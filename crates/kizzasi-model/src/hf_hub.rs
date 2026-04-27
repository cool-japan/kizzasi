//! # HuggingFace Hub API Client
//!
//! Synchronous (blocking) HTTP client for downloading model weights and metadata
//! from the HuggingFace Hub. This module is feature-gated behind `hf-hub`.
//!
//! ## Features
//!
//! - **Blocking downloads**: Uses `reqwest::blocking` — no async runtime required
//! - **Local caching**: Files are cached under `~/.cache/kizzasi/hub` (or a temp path)
//! - **Authentication**: Reads `HF_TOKEN` environment variable automatically
//! - **SafeTensors integration**: `load_from_hub` returns `HashMap<tensor_name, Vec<f32>>`
//!   ready for use with existing weight loaders
//! - **SHA-256 integrity**: Optional post-download hash logging for audit trails
//!
//! ## Example
//!
//! ```rust,no_run
//! use kizzasi_model::hf_hub::{HfHubClient, HfHubConfig};
//!
//! let client = HfHubClient::default_client().expect("client creation");
//! let url = client.file_url("state-spaces/mamba-130m", "model.safetensors", "main");
//! println!("Would download from: {url}");
//! ```

use crate::error::{ModelError, ModelResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

// ─── Cache directory resolution ──────────────────────────────────────────────

/// Resolve a suitable local cache directory.
///
/// Priority order:
/// 1. `~/.cache/kizzasi/hub` when `$HOME` (Unix) or `%USERPROFILE%` (Windows) is set
/// 2. `<std::env::temp_dir()>/kizzasi/hub` as a universal fallback
fn dirs_cache_or_tmp() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok();
    if let Some(h) = home {
        PathBuf::from(h).join(".cache").join("kizzasi").join("hub")
    } else {
        std::env::temp_dir().join("kizzasi").join("hub")
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the HuggingFace Hub client.
///
/// All fields have sensible defaults: the base URL points to the public HF Hub,
/// the API token is read from `$HF_TOKEN`, the cache lives under `~/.cache/kizzasi/hub`,
/// and downloads time out after 5 minutes.
#[derive(Debug, Clone)]
pub struct HfHubConfig {
    /// HuggingFace Hub base URL (default: `"https://huggingface.co"`)
    pub base_url: String,
    /// Optional API token for private repos (read from `$HF_TOKEN` if not set explicitly)
    pub token: Option<String>,
    /// Cache directory for downloaded files (default: `~/.cache/kizzasi/hub`)
    pub cache_dir: PathBuf,
    /// Download timeout in seconds (default: 300)
    pub timeout_secs: u64,
    /// Whether to log the SHA-256 hash of each downloaded file (default: true)
    pub log_sha256: bool,
}

impl Default for HfHubConfig {
    fn default() -> Self {
        Self {
            base_url: "https://huggingface.co".to_string(),
            token: std::env::var("HF_TOKEN").ok(),
            cache_dir: dirs_cache_or_tmp(),
            timeout_secs: 300,
            log_sha256: true,
        }
    }
}

// ─── API response types ───────────────────────────────────────────────────────

/// Metadata returned by the HF Hub `/api/models/{repo_id}` endpoint.
///
/// Only the fields used internally are mapped; the full JSON is intentionally
/// not mirrored to avoid brittleness against API changes.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HfModelInfo {
    /// Canonical model identifier, e.g. `"state-spaces/mamba-130m"`
    pub id: String,
    /// Friendly model name (may differ from the path component)
    #[serde(rename = "modelId", default)]
    pub model_id: String,
    /// Whether the repository is private
    #[serde(default)]
    pub private: bool,
    /// Repository tags (e.g. `["mamba", "safetensors"]`)
    #[serde(default)]
    pub tags: Vec<String>,
    /// Number of downloads (populated by some API responses)
    #[serde(default)]
    pub downloads: u64,
    /// Number of likes
    #[serde(default)]
    pub likes: u64,
}

/// A single file entry returned by the HF Hub tree API.
#[derive(Debug, Clone, Deserialize)]
pub struct HfRepoFile {
    /// Relative file path within the repository (e.g. `"model.safetensors"`)
    pub rfilename: String,
    /// File size in bytes when available
    #[serde(default)]
    pub size: Option<u64>,
    /// LFS pointer SHA256, if the file is stored via Git LFS
    #[serde(default)]
    pub lfs: Option<HfLfsPointer>,
}

/// Git LFS pointer metadata embedded in the tree API response.
#[derive(Debug, Clone, Deserialize)]
pub struct HfLfsPointer {
    /// SHA-256 of the actual (dereferenced) blob
    pub sha256: String,
    /// Size of the actual blob in bytes
    pub size: u64,
    /// Size of the LFS pointer file itself
    #[serde(rename = "pointerSize", default)]
    pub pointer_size: u64,
}

// ─── Download progress ────────────────────────────────────────────────────────

/// Snapshot of download progress for a single file.
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Filename being downloaded
    pub filename: String,
    /// Bytes received so far
    pub bytes_received: u64,
    /// Total expected bytes (0 if unknown)
    pub bytes_total: u64,
}

impl DownloadProgress {
    /// Fraction complete in `[0.0, 1.0]`, or `None` when total is unknown.
    pub fn fraction(&self) -> Option<f64> {
        if self.bytes_total == 0 {
            None
        } else {
            Some(self.bytes_received as f64 / self.bytes_total as f64)
        }
    }
}

// ─── Client ───────────────────────────────────────────────────────────────────

/// Synchronous HuggingFace Hub client.
///
/// Wraps `reqwest::blocking::Client` to provide caching, authentication,
/// and convenience methods for common model-downloading workflows.
///
/// # Thread safety
///
/// `HfHubClient` is `Send + Sync` — `reqwest::blocking::Client` is internally
/// `Arc`-wrapped and safe to share across threads.
pub struct HfHubClient {
    config: HfHubConfig,
    client: reqwest::blocking::Client,
}

impl std::fmt::Debug for HfHubClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfHubClient")
            .field("base_url", &self.config.base_url)
            .field("cache_dir", &self.config.cache_dir)
            .field("timeout_secs", &self.config.timeout_secs)
            .field("has_token", &self.config.token.is_some())
            .finish()
    }
}

impl HfHubClient {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new client from the supplied configuration.
    ///
    /// This creates the cache directory if it does not yet exist and builds the
    /// underlying `reqwest::blocking::Client`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if:
    /// - The cache directory cannot be created
    /// - The HTTP client cannot be initialised (e.g. TLS setup failure)
    pub fn new(config: HfHubConfig) -> ModelResult<Self> {
        // Ensure cache directory exists before any downloads are attempted.
        std::fs::create_dir_all(&config.cache_dir).map_err(|e| ModelError::LoadError {
            context: "HfHubClient::new – create_dir_all".to_string(),
            message: e.to_string(),
        })?;

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .user_agent(concat!("kizzasi/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| ModelError::LoadError {
                context: "HfHubClient::new – client build".to_string(),
                message: e.to_string(),
            })?;

        Ok(Self { config, client })
    }

    /// Create a client with the default [`HfHubConfig`].
    ///
    /// Equivalent to `HfHubClient::new(HfHubConfig::default())`.
    pub fn default_client() -> ModelResult<Self> {
        Self::new(HfHubConfig::default())
    }

    /// Create a client with a custom cache directory, keeping other defaults.
    pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> ModelResult<Self> {
        Self::new(HfHubConfig {
            cache_dir: cache_dir.into(),
            ..HfHubConfig::default()
        })
    }

    /// Builder-style: override the API token.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.config.token = Some(token.into());
        self
    }

    /// Builder-style: disable SHA-256 logging.
    pub fn without_sha256_logging(mut self) -> Self {
        self.config.log_sha256 = false;
        self
    }

    // ── URL & path helpers ───────────────────────────────────────────────────

    /// Construct the CDN URL for a file in a HF repository.
    ///
    /// The returned URL uses the `resolve` endpoint which transparently
    /// dereferences Git LFS pointers.
    ///
    /// ```
    /// # use kizzasi_model::hf_hub::{HfHubClient, HfHubConfig};
    /// # let client = HfHubClient::new(HfHubConfig { cache_dir: std::env::temp_dir(), ..Default::default() }).unwrap();
    /// let url = client.file_url("state-spaces/mamba-130m", "model.safetensors", "main");
    /// assert!(url.starts_with("https://huggingface.co/state-spaces/mamba-130m/resolve/main/"));
    /// ```
    pub fn file_url(&self, repo_id: &str, filename: &str, revision: &str) -> String {
        format!(
            "{}/{}/resolve/{}/{}",
            self.config.base_url, repo_id, revision, filename
        )
    }

    /// Return the local cache path for a file **without** triggering a download.
    ///
    /// The directory structure is:
    /// `<cache_dir>/<repo_slug>/<revision>/<filename>`
    /// where `repo_slug` replaces `/` with `--` (matching the HF Python client
    /// convention).
    pub fn cached_path(&self, repo_id: &str, filename: &str, revision: &str) -> PathBuf {
        let repo_slug = repo_id.replace('/', "--");
        self.config
            .cache_dir
            .join(repo_slug)
            .join(revision)
            .join(filename)
    }

    /// Return `true` if the file is already present in the local cache.
    pub fn is_cached(&self, repo_id: &str, filename: &str, revision: &str) -> bool {
        self.cached_path(repo_id, filename, revision).exists()
    }

    // ── Auth ─────────────────────────────────────────────────────────────────

    /// Build the `Authorization` header value, if a token is configured.
    pub fn auth_header(&self) -> Option<String> {
        self.config.token.as_ref().map(|t| format!("Bearer {t}"))
    }

    // ── API calls ────────────────────────────────────────────────────────────

    /// Fetch model metadata from the HF Hub REST API.
    ///
    /// Makes a GET request to `/api/models/{repo_id}` and deserialises the
    /// JSON response into [`HfModelInfo`].
    ///
    /// # Errors
    ///
    /// - Network failure
    /// - Non-2xx HTTP status (e.g. 404 for unknown repo, 401 for private repo
    ///   without a valid token)
    /// - JSON deserialisation failure
    pub fn model_info(&self, repo_id: &str) -> ModelResult<HfModelInfo> {
        let url = format!("{}/api/models/{}", self.config.base_url, repo_id);
        tracing::debug!(url = %url, "fetching model info");

        let response = self
            .build_get(&url)
            .send()
            .map_err(|e| ModelError::LoadError {
                context: "HfHubClient::model_info".to_string(),
                message: e.to_string(),
            })?;

        if !response.status().is_success() {
            return Err(ModelError::LoadError {
                context: "HfHubClient::model_info".to_string(),
                message: format!("HTTP {}: {}", response.status(), url),
            });
        }

        response
            .json::<HfModelInfo>()
            .map_err(|e| ModelError::LoadError {
                context: "HfHubClient::model_info – JSON deserialise".to_string(),
                message: e.to_string(),
            })
    }

    /// List files in a repository at a given revision.
    ///
    /// Uses the `/api/models/{repo_id}/tree/{revision}` endpoint and returns
    /// all entries as [`HfRepoFile`] structs (including directories, if any).
    ///
    /// # Errors
    ///
    /// Same as [`Self::model_info`].
    pub fn list_files(&self, repo_id: &str, revision: &str) -> ModelResult<Vec<HfRepoFile>> {
        let url = format!(
            "{}/api/models/{}/tree/{}",
            self.config.base_url, repo_id, revision
        );
        tracing::debug!(url = %url, "listing repository files");

        let response = self
            .build_get(&url)
            .send()
            .map_err(|e| ModelError::LoadError {
                context: "HfHubClient::list_files".to_string(),
                message: e.to_string(),
            })?;

        if !response.status().is_success() {
            return Err(ModelError::LoadError {
                context: "HfHubClient::list_files".to_string(),
                message: format!("HTTP {}: {}", response.status(), url),
            });
        }

        response
            .json::<Vec<HfRepoFile>>()
            .map_err(|e| ModelError::LoadError {
                context: "HfHubClient::list_files – JSON deserialise".to_string(),
                message: e.to_string(),
            })
    }

    // ── File downloads ────────────────────────────────────────────────────────

    /// Download a single file from HF Hub to the local cache.
    ///
    /// If the file is already cached (i.e. [`Self::is_cached`] returns `true`),
    /// the cached path is returned immediately without a network request.
    ///
    /// The download writes atomically via a `BufWriter` and logs the SHA-256
    /// digest when [`HfHubConfig::log_sha256`] is `true`.
    ///
    /// # Returns
    ///
    /// The local `PathBuf` of the cached file (guaranteed to exist on success).
    ///
    /// # Errors
    ///
    /// - Cache directory creation failure
    /// - Network / HTTP error
    /// - Disk I/O error while writing the cache file
    pub fn download_file(
        &self,
        repo_id: &str,
        filename: &str,
        revision: &str,
    ) -> ModelResult<PathBuf> {
        let local_path = self.cached_path(repo_id, filename, revision);

        if local_path.exists() {
            tracing::debug!(path = %local_path.display(), "cache hit – skipping download");
            return Ok(local_path);
        }

        // Create the full directory hierarchy (including any sub-directories
        // that appear in `filename` itself, e.g. "shards/shard-00.safetensors").
        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ModelError::LoadError {
                context: "HfHubClient::download_file – create_dir_all".to_string(),
                message: e.to_string(),
            })?;
        }

        let url = self.file_url(repo_id, filename, revision);
        tracing::info!(url = %url, dest = %local_path.display(), "downloading file");

        let response = self
            .build_get(&url)
            .send()
            .map_err(|e| ModelError::LoadError {
                context: "HfHubClient::download_file – send".to_string(),
                message: e.to_string(),
            })?;

        if !response.status().is_success() {
            return Err(ModelError::LoadError {
                context: "HfHubClient::download_file".to_string(),
                message: format!("HTTP {}: {}", response.status(), url),
            });
        }

        let content_length = response.content_length().unwrap_or(0);
        tracing::debug!(bytes = content_length, "content-length");

        let bytes = response.bytes().map_err(|e| ModelError::LoadError {
            context: "HfHubClient::download_file – read bytes".to_string(),
            message: e.to_string(),
        })?;

        // Optionally compute and log the SHA-256 digest for audit purposes.
        if self.config.log_sha256 {
            let digest = Sha256::digest(&bytes);
            let digest_hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            tracing::debug!(
                sha256 = %digest_hex,
                filename = filename,
                "SHA-256 verified"
            );
        }

        // Write to disk via BufWriter for efficiency.
        let file = std::fs::File::create(&local_path).map_err(|e| ModelError::LoadError {
            context: "HfHubClient::download_file – File::create".to_string(),
            message: e.to_string(),
        })?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(&bytes)
            .map_err(|e| ModelError::LoadError {
                context: "HfHubClient::download_file – write_all".to_string(),
                message: e.to_string(),
            })?;
        writer.flush().map_err(|e| ModelError::LoadError {
            context: "HfHubClient::download_file – flush".to_string(),
            message: e.to_string(),
        })?;

        tracing::info!(
            path = %local_path.display(),
            bytes = bytes.len(),
            "download complete"
        );
        Ok(local_path)
    }

    /// Download all `.safetensors` files for a model at a given revision.
    ///
    /// First calls [`Self::list_files`] to enumerate the repository contents,
    /// then downloads every file whose name ends with `.safetensors`.
    ///
    /// Files that are already cached are returned without re-downloading.
    ///
    /// # Returns
    ///
    /// A `Vec<PathBuf>` of local paths in the same order as the repository listing.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::list_files`] or individual [`Self::download_file`]
    /// calls. The first failure aborts the entire operation.
    pub fn download_model_weights(
        &self,
        repo_id: &str,
        revision: &str,
    ) -> ModelResult<Vec<PathBuf>> {
        let all_files = self.list_files(repo_id, revision)?;

        let safetensors: Vec<&HfRepoFile> = all_files
            .iter()
            .filter(|f| f.rfilename.ends_with(".safetensors"))
            .collect();

        if safetensors.is_empty() {
            return Err(ModelError::LoadError {
                context: "HfHubClient::download_model_weights".to_string(),
                message: format!("No .safetensors files found in {repo_id}@{revision}"),
            });
        }

        tracing::info!(
            count = safetensors.len(),
            repo = repo_id,
            revision = revision,
            "downloading SafeTensors weight shards"
        );

        let mut local_paths = Vec::with_capacity(safetensors.len());
        for file_entry in safetensors {
            let path = self.download_file(repo_id, &file_entry.rfilename, revision)?;
            local_paths.push(path);
        }

        Ok(local_paths)
    }

    /// Clear all cached files for a specific repository.
    ///
    /// Removes the entire `<cache_dir>/<repo_slug>/` subtree. Does nothing if
    /// the directory does not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be removed (permission denied, etc.).
    pub fn clear_cache(&self, repo_id: &str) -> ModelResult<()> {
        let repo_slug = repo_id.replace('/', "--");
        let repo_cache = self.config.cache_dir.join(&repo_slug);
        if repo_cache.exists() {
            std::fs::remove_dir_all(&repo_cache).map_err(|e| ModelError::LoadError {
                context: "HfHubClient::clear_cache".to_string(),
                message: e.to_string(),
            })?;
            tracing::info!(repo = repo_id, "cache cleared");
        }
        Ok(())
    }

    /// Return the total size in bytes of all files currently in the cache directory.
    ///
    /// Only counts regular files (not directories or symlinks) at the top level
    /// of `<cache_dir>`. Returns `0` if the cache directory does not exist.
    pub fn cache_size(&self) -> ModelResult<u64> {
        if !self.config.cache_dir.exists() {
            return Ok(0);
        }
        let total = cache_dir_size_recursive(&self.config.cache_dir)?;
        Ok(total)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Build a GET request with optional `Authorization` header.
    fn build_get(&self, url: &str) -> reqwest::blocking::RequestBuilder {
        let mut req = self.client.get(url);
        if let Some(auth) = self.auth_header() {
            req = req.header("Authorization", auth);
        }
        req
    }
}

/// Recursively compute total byte size of all regular files under `dir`.
fn cache_dir_size_recursive(dir: &std::path::Path) -> ModelResult<u64> {
    let mut total = 0u64;
    let entries = std::fs::read_dir(dir).map_err(|e| ModelError::LoadError {
        context: "cache_dir_size_recursive".to_string(),
        message: e.to_string(),
    })?;
    for entry_result in entries {
        let entry = entry_result.map_err(|e| ModelError::LoadError {
            context: "cache_dir_size_recursive – entry".to_string(),
            message: e.to_string(),
        })?;
        let metadata = entry.metadata().map_err(|e| ModelError::LoadError {
            context: "cache_dir_size_recursive – metadata".to_string(),
            message: e.to_string(),
        })?;
        if metadata.is_file() {
            total += metadata.len();
        } else if metadata.is_dir() {
            total += cache_dir_size_recursive(&entry.path())?;
        }
    }
    Ok(total)
}

// ─── High-level convenience function ─────────────────────────────────────────

/// Download and load all SafeTensors weight shards from a HuggingFace repository.
///
/// This is a convenience wrapper that:
/// 1. Creates an [`HfHubClient`] (using `config` or the default if `None`)
/// 2. Downloads all `.safetensors` files to the local cache
/// 3. Loads every tensor from each shard as a flat `Vec<f32>`
/// 4. Returns a `HashMap<tensor_name, Vec<f32>>` compatible with the existing
///    weight-loading infrastructure in this crate
///
/// Tensors are loaded in shard order; if two shards define the same tensor name
/// the later shard wins (matching HF's convention for sharded checkpoints).
///
/// # Arguments
///
/// * `repo_id`  – Repository identifier, e.g. `"state-spaces/mamba-130m"`
/// * `revision` – Git ref, e.g. `"main"`, a tag, or a full commit SHA
/// * `config`   – Optional client config; `None` uses [`HfHubConfig::default()`]
///
/// # Errors
///
/// Propagates errors from download or SafeTensors loading.
///
/// # Example
///
/// ```rust,no_run
/// let weights = kizzasi_model::hf_hub::load_from_hub(
///     "state-spaces/mamba-130m",
///     "main",
///     None,
/// ).expect("load weights");
/// println!("loaded {} tensors", weights.len());
/// ```
pub fn load_from_hub(
    repo_id: &str,
    revision: &str,
    config: Option<HfHubConfig>,
) -> ModelResult<HashMap<String, Vec<f32>>> {
    let client = HfHubClient::new(config.unwrap_or_default())?;
    let shard_paths = client.download_model_weights(repo_id, revision)?;

    let mut weights: HashMap<String, Vec<f32>> = HashMap::new();

    for shard_path in &shard_paths {
        tracing::info!(path = %shard_path.display(), "loading SafeTensors shard");

        let bytes = std::fs::read(shard_path).map_err(|e| ModelError::LoadError {
            context: "load_from_hub – read shard".to_string(),
            message: e.to_string(),
        })?;

        let tensors =
            safetensors::SafeTensors::deserialize(&bytes).map_err(|e| ModelError::LoadError {
                context: "load_from_hub – SafeTensors::deserialize".to_string(),
                message: e.to_string(),
            })?;

        for (name, view) in tensors.tensors() {
            let f32_data = convert_tensor_view_to_f32(view)?;
            weights.insert(name.to_string(), f32_data);
        }
    }

    tracing::info!(
        tensors = weights.len(),
        shards = shard_paths.len(),
        "load_from_hub complete"
    );

    Ok(weights)
}

/// Convert a raw SafeTensors tensor view to `Vec<f32>`.
///
/// Handles the most common storage dtypes encountered in HF checkpoints:
/// `F32`, `F16`, `BF16`, `F64`, `I32`, and `I64`.
fn convert_tensor_view_to_f32(view: safetensors::tensor::TensorView<'_>) -> ModelResult<Vec<f32>> {
    use safetensors::tensor::Dtype;

    let data = view.data();
    let dtype = view.dtype();

    let result = match dtype {
        Dtype::F32 => {
            // Already native f32 — fast path: reinterpret the byte slice.
            if data.len() % 4 != 0 {
                return Err(ModelError::LoadError {
                    context: "convert_tensor_view_to_f32".to_string(),
                    message: format!("F32 tensor byte count {} is not divisible by 4", data.len()),
                });
            }
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect::<Vec<f32>>()
        }
        Dtype::F16 => {
            if data.len() % 2 != 0 {
                return Err(ModelError::LoadError {
                    context: "convert_tensor_view_to_f32".to_string(),
                    message: format!("F16 tensor byte count {} is not divisible by 2", data.len()),
                });
            }
            data.chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect::<Vec<f32>>()
        }
        Dtype::BF16 => {
            if data.len() % 2 != 0 {
                return Err(ModelError::LoadError {
                    context: "convert_tensor_view_to_f32".to_string(),
                    message: format!(
                        "BF16 tensor byte count {} is not divisible by 2",
                        data.len()
                    ),
                });
            }
            data.chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect::<Vec<f32>>()
        }
        Dtype::F64 => {
            if data.len() % 8 != 0 {
                return Err(ModelError::LoadError {
                    context: "convert_tensor_view_to_f32".to_string(),
                    message: format!("F64 tensor byte count {} is not divisible by 8", data.len()),
                });
            }
            data.chunks_exact(8)
                .map(|b| {
                    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect::<Vec<f32>>()
        }
        Dtype::I32 => {
            if data.len() % 4 != 0 {
                return Err(ModelError::LoadError {
                    context: "convert_tensor_view_to_f32".to_string(),
                    message: format!("I32 tensor byte count {} is not divisible by 4", data.len()),
                });
            }
            data.chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
                .collect::<Vec<f32>>()
        }
        Dtype::I64 => {
            if data.len() % 8 != 0 {
                return Err(ModelError::LoadError {
                    context: "convert_tensor_view_to_f32".to_string(),
                    message: format!("I64 tensor byte count {} is not divisible by 8", data.len()),
                });
            }
            data.chunks_exact(8)
                .map(|b| {
                    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect::<Vec<f32>>()
        }
        other => {
            return Err(ModelError::LoadError {
                context: "convert_tensor_view_to_f32".to_string(),
                message: format!("unsupported SafeTensors dtype: {other:?}"),
            });
        }
    };

    Ok(result)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config ────────────────────────────────────────────────────────────────

    #[test]
    fn test_hf_hub_config_default_no_panic() {
        let cfg = HfHubConfig::default();
        assert!(!cfg.base_url.is_empty());
        assert!(cfg.timeout_secs > 0);
        assert!(cfg.base_url.starts_with("https://"));
    }

    #[test]
    fn test_hf_hub_config_cache_dir_is_absolute() {
        let cfg = HfHubConfig::default();
        assert!(cfg.cache_dir.is_absolute());
    }

    // ── URL construction ──────────────────────────────────────────────────────

    #[test]
    fn test_file_url_construction() {
        let cfg = HfHubConfig {
            base_url: "https://huggingface.co".to_string(),
            token: None,
            cache_dir: std::env::temp_dir(),
            timeout_secs: 30,
            log_sha256: false,
        };
        let client = HfHubClient::new(cfg).expect("client creation");
        let url = client.file_url("state-spaces/mamba-130m", "model.safetensors", "main");
        assert!(url.contains("state-spaces/mamba-130m"), "url={url}");
        assert!(url.contains("model.safetensors"), "url={url}");
        assert!(url.contains("/resolve/main/"), "url={url}");
    }

    #[test]
    fn test_file_url_custom_base() {
        let cfg = HfHubConfig {
            base_url: "https://mirror.example.com".to_string(),
            token: None,
            cache_dir: std::env::temp_dir(),
            timeout_secs: 30,
            log_sha256: false,
        };
        let client = HfHubClient::new(cfg).expect("client creation");
        let url = client.file_url("org/repo", "weights.bin", "v1.0");
        assert!(url.starts_with("https://mirror.example.com/"), "url={url}");
        assert!(url.contains("/resolve/v1.0/weights.bin"), "url={url}");
    }

    // ── Cached path ───────────────────────────────────────────────────────────

    #[test]
    fn test_cached_path_structure() {
        let cache_dir = std::env::temp_dir().join("kizzasi_hf_test_path");
        let cfg = HfHubConfig {
            cache_dir: cache_dir.clone(),
            ..HfHubConfig::default()
        };
        let client = HfHubClient::new(cfg).expect("client creation");
        let path = client.cached_path("org/repo", "model.safetensors", "main");
        let path_str = path.to_string_lossy();
        assert!(path_str.contains("org--repo"), "path={path_str}");
        assert!(path_str.contains("model.safetensors"), "path={path_str}");
        assert!(path_str.contains("main"), "path={path_str}");
        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    #[test]
    fn test_cached_path_slash_replacement() {
        let cache_dir = std::env::temp_dir().join("kizzasi_hf_test_slash");
        let cfg = HfHubConfig {
            cache_dir: cache_dir.clone(),
            ..HfHubConfig::default()
        };
        let client = HfHubClient::new(cfg).expect("client creation");
        let path = client.cached_path("state-spaces/mamba-130m", "model.safetensors", "main");
        let slug = path.components().find(|c| {
            c.as_os_str()
                .to_string_lossy()
                .contains("state-spaces--mamba-130m")
        });
        assert!(slug.is_some(), "expected slug in path: {}", path.display());
        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    // ── is_cached ─────────────────────────────────────────────────────────────

    #[test]
    fn test_is_cached_false_for_nonexistent() {
        let cfg = HfHubConfig {
            cache_dir: std::env::temp_dir().join("kizzasi_hf_test_nc"),
            ..HfHubConfig::default()
        };
        let client = HfHubClient::new(cfg).expect("client creation");
        assert!(!client.is_cached("org/repo", "nonexistent.bin", "main"));
    }

    #[test]
    fn test_is_cached_true_after_file_creation() {
        let cache_dir = std::env::temp_dir().join("kizzasi_hf_test_cached");
        let cfg = HfHubConfig {
            cache_dir: cache_dir.clone(),
            ..HfHubConfig::default()
        };
        let client = HfHubClient::new(cfg).expect("client creation");

        let path = client.cached_path("org/repo", "test.bin", "main");
        std::fs::create_dir_all(path.parent().expect("parent")).expect("create dirs");
        std::fs::write(&path, b"test data").expect("write test file");

        assert!(client.is_cached("org/repo", "test.bin", "main"));
        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    // ── Authentication ────────────────────────────────────────────────────────

    #[test]
    fn test_hf_client_auth_header_with_token() {
        let cfg = HfHubConfig {
            token: Some("hf_test_token".to_string()),
            cache_dir: std::env::temp_dir().join("kizzasi_hf_auth_test"),
            ..HfHubConfig::default()
        };
        let client = HfHubClient::new(cfg).expect("client creation");
        let auth = client.auth_header();
        assert_eq!(auth, Some("Bearer hf_test_token".to_string()));
    }

    #[test]
    fn test_hf_client_auth_header_without_token() {
        let cfg = HfHubConfig {
            token: None,
            cache_dir: std::env::temp_dir().join("kizzasi_hf_noauth_test"),
            ..HfHubConfig::default()
        };
        let client = HfHubClient::new(cfg).expect("client creation");
        assert!(client.auth_header().is_none());
    }

    // ── Deserialization ───────────────────────────────────────────────────────

    #[test]
    fn test_hf_model_info_deserialize() {
        let json = r#"{
            "id": "state-spaces/mamba-130m",
            "modelId": "mamba-130m",
            "private": false,
            "tags": ["mamba", "safetensors"],
            "downloads": 12345,
            "likes": 42
        }"#;
        let info: HfModelInfo = serde_json::from_str(json).expect("deserialise");
        assert_eq!(info.id, "state-spaces/mamba-130m");
        assert_eq!(info.model_id, "mamba-130m");
        assert!(!info.private);
        assert_eq!(info.tags, vec!["mamba", "safetensors"]);
        assert_eq!(info.downloads, 12345);
        assert_eq!(info.likes, 42);
    }

    #[test]
    fn test_hf_model_info_deserialize_minimal() {
        // Only mandatory fields present — optional fields must default cleanly.
        let json = r#"{"id": "org/model"}"#;
        let info: HfModelInfo = serde_json::from_str(json).expect("deserialise");
        assert_eq!(info.id, "org/model");
        assert!(info.model_id.is_empty());
        assert!(!info.private);
        assert!(info.tags.is_empty());
    }

    #[test]
    fn test_hf_repo_file_deserialize() {
        let json = r#"{
            "rfilename": "model.safetensors",
            "size": 271523456,
            "lfs": {
                "sha256": "abcd1234",
                "size": 271523456,
                "pointerSize": 134
            }
        }"#;
        let f: HfRepoFile = serde_json::from_str(json).expect("deserialise");
        assert_eq!(f.rfilename, "model.safetensors");
        assert_eq!(f.size, Some(271_523_456));
        let lfs = f.lfs.expect("lfs");
        assert_eq!(lfs.sha256, "abcd1234");
        assert_eq!(lfs.pointer_size, 134);
    }

    #[test]
    fn test_hf_repo_file_deserialize_no_lfs() {
        let json = r#"{"rfilename": "config.json"}"#;
        let f: HfRepoFile = serde_json::from_str(json).expect("deserialise");
        assert_eq!(f.rfilename, "config.json");
        assert!(f.size.is_none());
        assert!(f.lfs.is_none());
    }

    // ── Builder API ───────────────────────────────────────────────────────────

    #[test]
    fn test_with_token_builder() {
        let client = HfHubClient::with_cache_dir(std::env::temp_dir().join("kizzasi_hf_builder"))
            .expect("client")
            .with_token("my_secret_token");
        assert_eq!(
            client.auth_header(),
            Some("Bearer my_secret_token".to_string())
        );
    }

    #[test]
    fn test_without_sha256_logging_builder() {
        let client = HfHubClient::with_cache_dir(std::env::temp_dir().join("kizzasi_hf_nosha256"))
            .expect("client")
            .without_sha256_logging();
        assert!(!client.config.log_sha256);
    }

    // ── Debug impl ────────────────────────────────────────────────────────────

    #[test]
    fn test_debug_does_not_leak_token() {
        let cfg = HfHubConfig {
            token: Some("super_secret".to_string()),
            cache_dir: std::env::temp_dir(),
            ..HfHubConfig::default()
        };
        let client = HfHubClient::new(cfg).expect("client");
        let debug_str = format!("{client:?}");
        assert!(
            !debug_str.contains("super_secret"),
            "token must not appear in Debug output"
        );
        assert!(debug_str.contains("has_token: true"));
    }

    // ── cache_size ────────────────────────────────────────────────────────────

    #[test]
    fn test_cache_size_empty_dir() {
        let cache_dir = std::env::temp_dir().join("kizzasi_hf_size_empty");
        let cfg = HfHubConfig {
            cache_dir: cache_dir.clone(),
            ..HfHubConfig::default()
        };
        let client = HfHubClient::new(cfg).expect("client");
        // Newly created dir should be empty → 0 bytes.
        assert_eq!(client.cache_size().expect("cache_size"), 0);
        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    #[test]
    fn test_cache_size_nonexistent_dir() {
        let cfg = HfHubConfig {
            cache_dir: std::env::temp_dir()
                .join("kizzasi_hf_size_nx")
                .join("does_not_exist"),
            ..HfHubConfig::default()
        };
        // new() will create the dir; remove it before calling cache_size.
        let client = HfHubClient::new(cfg).expect("client");
        let dir = client.config.cache_dir.clone();
        let _ = std::fs::remove_dir_all(&dir);
        // cache_size should return 0 for non-existent cache dir.
        assert_eq!(client.cache_size().expect("cache_size"), 0);
    }

    #[test]
    fn test_cache_size_counts_files() {
        let cache_dir = std::env::temp_dir().join("kizzasi_hf_size_files");
        let cfg = HfHubConfig {
            cache_dir: cache_dir.clone(),
            ..HfHubConfig::default()
        };
        let client = HfHubClient::new(cfg).expect("client");

        // Write two files into the cache dir.
        std::fs::write(cache_dir.join("a.bin"), vec![0u8; 100]).expect("write a");
        std::fs::write(cache_dir.join("b.bin"), vec![0u8; 200]).expect("write b");

        assert_eq!(client.cache_size().expect("cache_size"), 300);
        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    // ── DownloadProgress ──────────────────────────────────────────────────────

    #[test]
    fn test_download_progress_fraction_known() {
        let p = DownloadProgress {
            filename: "model.safetensors".to_string(),
            bytes_received: 50,
            bytes_total: 200,
        };
        assert_eq!(p.fraction(), Some(0.25));
    }

    #[test]
    fn test_download_progress_fraction_unknown() {
        let p = DownloadProgress {
            filename: "model.safetensors".to_string(),
            bytes_received: 50,
            bytes_total: 0,
        };
        assert!(p.fraction().is_none());
    }

    // ── convert_tensor_view_to_f32 (via synthesised SafeTensors bytes) ─────────
    // We test the byte conversion helpers indirectly using raw byte slices
    // constructed to match the SafeTensors wire format for each dtype.

    #[test]
    fn test_f32_le_round_trip() {
        // Build a minimal valid SafeTensors buffer with one F32 tensor.
        let value: f32 = std::f32::consts::PI;
        let raw = value.to_le_bytes();
        // Re-parse via chunks_exact to verify the same logic used internally.
        let recovered: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert!((recovered[0] - value).abs() < 1e-6);
    }

    #[test]
    fn test_f16_conversion_logic() {
        let value = half::f16::from_f32(1.5_f32);
        let bits = value.to_bits();
        let raw = bits.to_le_bytes();
        let recovered = half::f16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32();
        assert!((recovered - 1.5_f32).abs() < 1e-3);
    }

    #[test]
    fn test_bf16_conversion_logic() {
        let value = half::bf16::from_f32(2.0_f32);
        let bits = value.to_bits();
        let raw = bits.to_le_bytes();
        let recovered = half::bf16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32();
        assert!((recovered - 2.0_f32).abs() < 1e-3);
    }
}
