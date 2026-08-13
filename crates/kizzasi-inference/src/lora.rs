//! LoRA (Low-Rank Adaptation) adapter loading for inference
//!
//! This module provides support for loading and applying LoRA adapters
//! at inference time, allowing efficient model fine-tuning and adaptation.

use crate::error::{InferenceError, InferenceResult};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// LoRA adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Rank of the low-rank matrices
    pub rank: usize,
    /// Scaling factor (alpha / rank)
    pub alpha: f32,
    /// Dropout rate for LoRA layers
    pub dropout: f32,
    /// Target modules to apply LoRA to
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        }
    }
}

impl LoraConfig {
    /// Create a new LoRA configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the rank
    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    /// Set alpha (scaling factor)
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set dropout rate
    pub fn dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Add target module
    pub fn add_target_module(mut self, module: impl Into<String>) -> Self {
        self.target_modules.push(module.into());
        self
    }

    /// Get the effective scaling factor
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

/// A LoRA adapter consisting of two low-rank matrices
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Low-rank matrix A (rank × in_features)
    pub lora_a: Array2<f32>,
    /// Low-rank matrix B (out_features × rank)
    pub lora_b: Array2<f32>,
    /// Scaling factor
    pub scaling: f32,
    /// Adapter name/identifier
    pub name: String,
}

impl LoraAdapter {
    /// Create a new LoRA adapter
    pub fn new(
        lora_a: Array2<f32>,
        lora_b: Array2<f32>,
        scaling: f32,
        name: impl Into<String>,
    ) -> InferenceResult<Self> {
        // Validate dimensions: A is (rank, in_features), B is (out_features, rank)
        let rank_a = lora_a.nrows();
        let rank_b = lora_b.ncols();

        if rank_a != rank_b {
            return Err(InferenceError::DimensionMismatch {
                expected: rank_a,
                got: rank_b,
            });
        }

        Ok(Self {
            lora_a,
            lora_b,
            scaling,
            name: name.into(),
        })
    }

    /// Get the rank of this adapter
    pub fn rank(&self) -> usize {
        self.lora_a.nrows()
    }

    /// Get input features dimension
    pub fn in_features(&self) -> usize {
        self.lora_a.ncols()
    }

    /// Get output features dimension
    pub fn out_features(&self) -> usize {
        self.lora_b.nrows()
    }

    /// Apply the LoRA adapter to an input, returning the scaled low-rank
    /// delta `scaling * (input @ A^T @ B^T)`.
    ///
    /// This is a *delta*, not a full layer output: LoRA's defining property
    /// is that it augments a frozen base layer (`W0 x + scaling * B A x`), so
    /// adding the base layer's own output is the caller's responsibility.
    /// The returned array always has [`LoraAdapter::out_features`] elements —
    /// unconditionally, regardless of `input`'s length, with no
    /// shape-dependent branching. Use [`LoraAdapter::apply_with_base`] when
    /// you have the base layer's output on hand and want the full
    /// `base + delta` sum, with a dimension check instead of a silently
    /// different formula.
    pub fn apply(&self, input: &Array1<f32>) -> InferenceResult<Array1<f32>> {
        if input.len() != self.in_features() {
            return Err(InferenceError::DimensionMismatch {
                expected: self.in_features(),
                got: input.len(),
            });
        }

        // Compute input @ A^T
        let mut hidden = Array1::zeros(self.rank());
        for i in 0..self.rank() {
            hidden[i] = input.dot(&self.lora_a.row(i));
        }

        // Compute hidden @ B^T
        let mut output = Array1::zeros(self.out_features());
        for i in 0..self.out_features() {
            output[i] = hidden.dot(&self.lora_b.row(i));
        }

        Ok(&output * self.scaling)
    }

    /// Apply the adapter to `input` and add the result to `base_output`.
    ///
    /// Computes the full LoRA composition `base_output + scaling * B A x`.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::DimensionMismatch`] when `base_output.len()`
    /// does not equal [`LoraAdapter::out_features`], instead of silently
    /// falling back to the bare delta.
    pub fn apply_with_base(
        &self,
        input: &Array1<f32>,
        base_output: &Array1<f32>,
    ) -> InferenceResult<Array1<f32>> {
        let delta = self.apply(input)?;
        if base_output.len() != delta.len() {
            return Err(InferenceError::DimensionMismatch {
                expected: delta.len(),
                got: base_output.len(),
            });
        }
        Ok(&delta + base_output)
    }

    /// Apply the LoRA adapter to a batch of inputs
    ///
    /// An empty batch yields an empty `(0, out_features)` array; the row width is
    /// taken from the adapter rather than from the first output row, which does not
    /// exist in that case.
    pub fn apply_batch(&self, inputs: &Array2<f32>) -> InferenceResult<Array2<f32>> {
        let batch_size = inputs.nrows();

        // An `Array2<f32>` with zero rows is legally constructible; indexing
        // `outputs[0]` for the row width would panic on it.
        if batch_size == 0 {
            return Ok(Array2::zeros((0, self.out_features())));
        }

        if inputs.ncols() != self.in_features() {
            return Err(InferenceError::DimensionMismatch {
                expected: self.in_features(),
                got: inputs.ncols(),
            });
        }

        let out_dim = self.out_features();
        let mut flat: Vec<f32> = Vec::with_capacity(batch_size * out_dim);

        for i in 0..batch_size {
            let input_row = inputs.row(i).to_owned();
            let output_row = self.apply(&input_row)?;
            flat.extend(output_row.iter().copied());
        }

        Array2::from_shape_vec((batch_size, out_dim), flat).map_err(|e| {
            InferenceError::ForwardError(format!("Failed to stack LoRA outputs: {}", e))
        })
    }
}

/// Manager for multiple LoRA adapters
pub struct LoraAdapterManager {
    /// Map of adapter names to adapters
    adapters: HashMap<String, LoraAdapter>,
    /// Active adapter name (if any)
    active_adapter: Option<String>,
    /// Configuration
    config: LoraConfig,
}

impl LoraAdapterManager {
    /// Create a new adapter manager
    pub fn new(config: LoraConfig) -> Self {
        Self {
            adapters: HashMap::new(),
            active_adapter: None,
            config,
        }
    }

    /// Register a new adapter
    pub fn register_adapter(&mut self, adapter: LoraAdapter) {
        let name = adapter.name.clone();
        self.adapters.insert(name, adapter);
    }

    /// Activate an adapter by name
    pub fn activate(&mut self, name: impl AsRef<str>) -> InferenceResult<()> {
        let name_ref = name.as_ref();
        if !self.adapters.contains_key(name_ref) {
            return Err(InferenceError::ForwardError(format!(
                "Adapter '{}' not found",
                name_ref
            )));
        }
        self.active_adapter = Some(name_ref.to_string());
        Ok(())
    }

    /// Deactivate the current adapter
    pub fn deactivate(&mut self) {
        self.active_adapter = None;
    }

    /// Get the active adapter
    pub fn active_adapter(&self) -> Option<&LoraAdapter> {
        self.active_adapter
            .as_ref()
            .and_then(|name| self.adapters.get(name))
    }

    /// Apply the active adapter (if any) to `base_output`, adding its scaled
    /// delta on top via [`LoraAdapter::apply_with_base`]. Returns
    /// `base_output` unchanged when no adapter is active.
    pub fn apply(&self, base_output: &Array1<f32>) -> InferenceResult<Array1<f32>> {
        if let Some(adapter) = self.active_adapter() {
            adapter.apply_with_base(base_output, base_output)
        } else {
            // No active adapter, return input unchanged
            Ok(base_output.clone())
        }
    }

    /// Apply the active adapter to a batch
    pub fn apply_batch(&self, inputs: &Array2<f32>) -> InferenceResult<Array2<f32>> {
        if let Some(adapter) = self.active_adapter() {
            adapter.apply_batch(inputs)
        } else {
            Ok(inputs.clone())
        }
    }

    /// List all registered adapters
    pub fn list_adapters(&self) -> Vec<&String> {
        self.adapters.keys().collect()
    }

    /// Get adapter by name
    pub fn get_adapter(&self, name: impl AsRef<str>) -> Option<&LoraAdapter> {
        self.adapters.get(name.as_ref())
    }

    /// Remove an adapter
    pub fn remove_adapter(&mut self, name: impl AsRef<str>) -> Option<LoraAdapter> {
        let name_ref = name.as_ref();
        // Deactivate if it's the active one
        if self.active_adapter.as_deref() == Some(name_ref) {
            self.deactivate();
        }
        self.adapters.remove(name_ref)
    }

    /// Get configuration
    pub fn config(&self) -> &LoraConfig {
        &self.config
    }
}

/// Builder for creating LoRA adapters from components
pub struct LoraAdapterBuilder {
    lora_a: Option<Array2<f32>>,
    lora_b: Option<Array2<f32>>,
    scaling: f32,
    name: String,
}

impl LoraAdapterBuilder {
    /// Create a new builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            lora_a: None,
            lora_b: None,
            scaling: 1.0,
            name: name.into(),
        }
    }

    /// Set matrix A
    pub fn lora_a(mut self, matrix: Array2<f32>) -> Self {
        self.lora_a = Some(matrix);
        self
    }

    /// Set matrix B
    pub fn lora_b(mut self, matrix: Array2<f32>) -> Self {
        self.lora_b = Some(matrix);
        self
    }

    /// Set scaling factor
    pub fn scaling(mut self, scaling: f32) -> Self {
        self.scaling = scaling;
        self
    }

    /// Set scaling from config
    pub fn scaling_from_config(mut self, config: &LoraConfig) -> Self {
        self.scaling = config.scaling();
        self
    }

    /// Build the adapter
    pub fn build(self) -> InferenceResult<LoraAdapter> {
        let lora_a = self.lora_a.ok_or_else(|| {
            InferenceError::ForwardError("LoRA matrix A not provided".to_string())
        })?;
        let lora_b = self.lora_b.ok_or_else(|| {
            InferenceError::ForwardError("LoRA matrix B not provided".to_string())
        })?;

        LoraAdapter::new(lora_a, lora_b, self.scaling, self.name)
    }
}

/// Wire format for serializing/deserializing a 2-D weight matrix as JSON.
///
/// Format on disk:
/// ```json
/// {"shape": [rows, cols], "data": [[…row0…], […row1…], …]}
/// ```
#[derive(Debug, Serialize, Deserialize)]
struct MatrixJson {
    shape: [usize; 2],
    data: Vec<Vec<f32>>,
}

impl MatrixJson {
    /// Convert an `Array2<f32>` into the serialisable wire form.
    fn from_array(array: &Array2<f32>) -> Self {
        let rows = array.nrows();
        let data = (0..rows).map(|r| array.row(r).to_vec()).collect();
        Self {
            shape: [rows, array.ncols()],
            data,
        }
    }

    /// Reconstruct an `Array2<f32>` from the wire form, validating shape consistency.
    fn into_array(self) -> Result<Array2<f32>, String> {
        let [rows, cols] = self.shape;
        if self.data.len() != rows {
            return Err(format!(
                "shape declares {} rows but data has {} rows",
                rows,
                self.data.len()
            ));
        }
        let mut flat = Vec::with_capacity(rows * cols);
        for (r, row) in self.data.into_iter().enumerate() {
            if row.len() != cols {
                return Err(format!(
                    "shape declares {} cols but row {} has {} elements",
                    cols,
                    r,
                    row.len()
                ));
            }
            flat.extend_from_slice(&row);
        }
        Array2::from_shape_vec((rows, cols), flat)
            .map_err(|e| format!("failed to build Array2 from weight data: {}", e))
    }
}

/// LoRA adapter loader for reading from and writing to disk
pub struct LoraAdapterLoader {
    /// Base path for adapter files
    base_path: PathBuf,
}

impl LoraAdapterLoader {
    /// Create a new loader with base path
    pub fn new(base_path: impl AsRef<Path>) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
        }
    }

    /// Save an adapter to disk.
    ///
    /// Creates `<base_path>/<adapter_name>/` (and any missing parents) then
    /// writes three files:
    ///
    /// - `config.json`  — serialised `LoraConfig`
    /// - `lora_a.json`  — matrix A: `{"shape": [rank, in_features], "data": [[…]…]}`
    /// - `lora_b.json`  — matrix B: `{"shape": [out_features, rank], "data": [[…]…]}`
    pub fn save(
        &self,
        adapter_name: impl AsRef<str>,
        adapter: &LoraAdapter,
        config: &LoraConfig,
    ) -> InferenceResult<()> {
        let adapter_path = self.base_path.join(adapter_name.as_ref());
        std::fs::create_dir_all(&adapter_path)?;

        // Write config.json
        let config_json = serde_json::to_string_pretty(config).map_err(|e| {
            InferenceError::ForwardError(format!("Failed to serialise LoraConfig: {}", e))
        })?;
        std::fs::write(adapter_path.join("config.json"), config_json)?;

        // Write lora_a.json
        let a_json = serde_json::to_string_pretty(&MatrixJson::from_array(&adapter.lora_a))
            .map_err(|e| {
                InferenceError::ForwardError(format!("Failed to serialise lora_a: {}", e))
            })?;
        std::fs::write(adapter_path.join("lora_a.json"), a_json)?;

        // Write lora_b.json
        let b_json = serde_json::to_string_pretty(&MatrixJson::from_array(&adapter.lora_b))
            .map_err(|e| {
                InferenceError::ForwardError(format!("Failed to serialise lora_b: {}", e))
            })?;
        std::fs::write(adapter_path.join("lora_b.json"), b_json)?;

        Ok(())
    }

    /// Load an adapter from disk.
    ///
    /// Expected directory layout under `<base_path>/<adapter_name>/`:
    ///
    /// - `config.json`  — `LoraConfig` (optional; defaults used when absent)
    /// - `lora_a.json`  — matrix A: `{"shape": [rank, in_features], "data": [[…row…], …]}`
    /// - `lora_b.json`  — matrix B: `{"shape": [out_features, rank], "data": [[…row…], …]}`
    ///
    /// Returns `InferenceError::ForwardError` if either weight file is missing,
    /// unparseable, or contains shape inconsistencies.
    pub fn load(
        &self,
        adapter_name: impl AsRef<str>,
    ) -> InferenceResult<(LoraAdapter, LoraConfig)> {
        let name = adapter_name.as_ref();
        let adapter_path = self.base_path.join(name);

        // ── config.json ───────────────────────────────────────────────────────
        let config_path = adapter_path.join("config.json");
        let config: LoraConfig = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
                InferenceError::ForwardError(format!(
                    "Failed to read config for adapter '{}': {}",
                    name, e
                ))
            })?;
            serde_json::from_str(&config_str).map_err(|e| {
                InferenceError::ForwardError(format!(
                    "Failed to parse config for adapter '{}': {}",
                    name, e
                ))
            })?
        } else {
            LoraConfig::default()
        };

        // ── lora_a.json ───────────────────────────────────────────────────────
        let a_path = adapter_path.join("lora_a.json");
        if !a_path.exists() {
            return Err(InferenceError::ForwardError(format!(
                "Weight file lora_a.json not found for adapter '{}'",
                name
            )));
        }
        let a_str = std::fs::read_to_string(&a_path).map_err(|e| {
            InferenceError::ForwardError(format!(
                "Failed to read lora_a.json for adapter '{}': {}",
                name, e
            ))
        })?;
        let a_wire: MatrixJson = serde_json::from_str(&a_str).map_err(|e| {
            InferenceError::ForwardError(format!(
                "Failed to parse lora_a.json for adapter '{}': {}",
                name, e
            ))
        })?;
        let lora_a = a_wire.into_array().map_err(|e| {
            InferenceError::ForwardError(format!(
                "Invalid lora_a.json for adapter '{}': {}",
                name, e
            ))
        })?;

        // ── lora_b.json ───────────────────────────────────────────────────────
        let b_path = adapter_path.join("lora_b.json");
        if !b_path.exists() {
            return Err(InferenceError::ForwardError(format!(
                "Weight file lora_b.json not found for adapter '{}'",
                name
            )));
        }
        let b_str = std::fs::read_to_string(&b_path).map_err(|e| {
            InferenceError::ForwardError(format!(
                "Failed to read lora_b.json for adapter '{}': {}",
                name, e
            ))
        })?;
        let b_wire: MatrixJson = serde_json::from_str(&b_str).map_err(|e| {
            InferenceError::ForwardError(format!(
                "Failed to parse lora_b.json for adapter '{}': {}",
                name, e
            ))
        })?;
        let lora_b = b_wire.into_array().map_err(|e| {
            InferenceError::ForwardError(format!(
                "Invalid lora_b.json for adapter '{}': {}",
                name, e
            ))
        })?;

        // ── assemble ──────────────────────────────────────────────────────────
        let scaling = config.scaling();
        let adapter = LoraAdapter::new(lora_a, lora_b, scaling, name)?;
        Ok((adapter, config))
    }

    /// List available adapters in the base path
    pub fn list_available(&self) -> InferenceResult<Vec<String>> {
        let mut adapters = Vec::new();

        let entries = std::fs::read_dir(&self.base_path).map_err(|e| {
            InferenceError::ForwardError(format!("Failed to read adapter directory: {}", e))
        })?;

        for entry in entries.flatten() {
            if entry.path().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    adapters.push(name.to_string());
                }
            }
        }

        Ok(adapters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: `apply_batch` read `outputs[0]` for the row width, which panics
    /// on the legally constructible zero-row input.
    #[test]
    fn test_apply_batch_empty_input() {
        let adapter = LoraAdapter::new(
            Array2::from_elem((2, 4), 0.1),
            Array2::from_elem((3, 2), 0.1),
            1.0,
            "empty",
        )
        .expect("adapter construction must succeed");

        let empty = Array2::<f32>::zeros((0, 4));
        let out = adapter
            .apply_batch(&empty)
            .expect("empty batch must be accepted");
        assert_eq!(out.shape(), &[0, 3]);
    }

    #[test]
    fn test_apply_batch_rejects_wrong_width() {
        let adapter = LoraAdapter::new(
            Array2::from_elem((2, 4), 0.1),
            Array2::from_elem((3, 2), 0.1),
            1.0,
            "narrow",
        )
        .expect("adapter construction must succeed");

        let wrong = Array2::<f32>::zeros((2, 5));
        assert!(matches!(
            adapter.apply_batch(&wrong),
            Err(InferenceError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_lora_config() {
        let config = LoraConfig::new().rank(16).alpha(32.0);

        assert_eq!(config.rank, 16);
        assert_eq!(config.alpha, 32.0);
        assert_eq!(config.scaling(), 2.0); // alpha / rank = 32 / 16
    }

    #[test]
    fn test_lora_adapter_creation() {
        let lora_a = Array2::from_shape_vec((4, 8), vec![1.0; 32]).unwrap();
        let lora_b = Array2::from_shape_vec((8, 4), vec![0.5; 32]).unwrap();

        let adapter = LoraAdapter::new(lora_a, lora_b, 0.5, "test").unwrap();

        assert_eq!(adapter.rank(), 4);
        assert_eq!(adapter.in_features(), 8);
        assert_eq!(adapter.out_features(), 8);
    }

    #[test]
    fn test_lora_adapter_dimension_mismatch() {
        let lora_a = Array2::from_shape_vec((4, 8), vec![1.0; 32]).unwrap();
        let lora_b = Array2::from_shape_vec((8, 5), vec![0.5; 40]).unwrap(); // Rank mismatch

        let result = LoraAdapter::new(lora_a, lora_b, 0.5, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_lora_adapter_apply() {
        let rank = 2;
        let in_features = 4;
        let out_features = 4;

        let lora_a = Array2::from_shape_vec((rank, in_features), vec![0.1; 8]).unwrap();
        let lora_b = Array2::from_shape_vec((out_features, rank), vec![0.2; 8]).unwrap();

        let adapter = LoraAdapter::new(lora_a, lora_b, 1.0, "test").unwrap();

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let output = adapter.apply(&input).unwrap();

        assert_eq!(output.len(), out_features);
        // Output should be input + LoRA modification
    }

    /// Regression: `apply()` used to silently add an identity residual only
    /// when `out_features == in_features`, so the *same* adapter's `apply()`
    /// had a different formula depending on shape. It must now always
    /// return just the scaled delta, and `apply_with_base` must be the
    /// (dimension-checked) way to add a base output on top.
    #[test]
    fn test_apply_is_pure_delta_regardless_of_shape() {
        // Square case: previously this silently added `input` to the result.
        let square = LoraAdapter::new(
            Array2::from_elem((2, 4), 0.1),
            Array2::from_elem((4, 2), 0.2),
            1.0,
            "square",
        )
        .unwrap();
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let delta = square.apply(&input).unwrap();

        // Manually compute the expected low-rank delta (no residual): both
        // A rows are uniform 0.1, so hidden[i] = 0.1 * input.sum() for each
        // of the 2 rank dimensions; both B rows are uniform 0.2, so each
        // output element = 0.2 * hidden.sum().
        let hidden_sum: f32 = 2.0 * (input.sum() * 0.1);
        let expected_each = hidden_sum * 0.2;
        for got in delta.iter() {
            assert!(
                (got - expected_each).abs() < 1e-5,
                "square-shaped apply() must not add an identity residual: got {got}, want {expected_each}"
            );
        }

        // apply_with_base explicitly composes base + delta.
        let base = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0]);
        let composed = square.apply_with_base(&input, &base).unwrap();
        for (c, (b, d)) in composed.iter().zip(base.iter().zip(delta.iter())) {
            assert!((c - (b + d)).abs() < 1e-6);
        }

        // Rectangular case: in_features != out_features. `apply` must still
        // succeed and return exactly `out_features` elements unconditionally
        // — no shape-dependent branching.
        let rect = LoraAdapter::new(
            Array2::from_elem((2, 4), 0.1),
            Array2::from_elem((6, 2), 0.2),
            1.0,
            "rect",
        )
        .unwrap();
        let rect_delta = rect.apply(&input).unwrap();
        assert_eq!(rect_delta.len(), 6);

        // apply_with_base must reject a base whose length doesn't match
        // out_features, rather than silently choosing a different formula.
        let mismatched_base = Array1::from_vec(vec![0.0; 4]);
        assert!(matches!(
            rect.apply_with_base(&input, &mismatched_base),
            Err(InferenceError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_lora_manager() {
        let config = LoraConfig::new();
        let mut manager = LoraAdapterManager::new(config);

        let lora_a = Array2::from_shape_vec((2, 4), vec![0.1; 8]).unwrap();
        let lora_b = Array2::from_shape_vec((4, 2), vec![0.2; 8]).unwrap();
        let adapter = LoraAdapter::new(lora_a, lora_b, 1.0, "adapter1").unwrap();

        manager.register_adapter(adapter);
        assert_eq!(manager.list_adapters().len(), 1);

        manager.activate("adapter1").unwrap();
        assert!(manager.active_adapter().is_some());

        manager.deactivate();
        assert!(manager.active_adapter().is_none());
    }

    #[test]
    fn test_lora_manager_apply_without_adapter() {
        let config = LoraConfig::new();
        let manager = LoraAdapterManager::new(config);

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let output = manager.apply(&input).unwrap();

        // Without adapter, output should equal input
        assert_eq!(output, input);
    }

    #[test]
    fn test_lora_builder() {
        let lora_a = Array2::from_shape_vec((2, 4), vec![0.1; 8]).unwrap();
        let lora_b = Array2::from_shape_vec((4, 2), vec![0.2; 8]).unwrap();

        let adapter = LoraAdapterBuilder::new("test")
            .lora_a(lora_a)
            .lora_b(lora_b)
            .scaling(0.5)
            .build()
            .unwrap();

        assert_eq!(adapter.name, "test");
        assert_eq!(adapter.scaling, 0.5);
    }

    #[test]
    fn test_lora_builder_missing_matrix() {
        let lora_a = Array2::from_shape_vec((2, 4), vec![0.1; 8]).unwrap();

        let result = LoraAdapterBuilder::new("test")
            .lora_a(lora_a)
            // Missing lora_b
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_lora_adapter_batch() {
        let lora_a = Array2::from_shape_vec((2, 4), vec![0.1; 8]).unwrap();
        let lora_b = Array2::from_shape_vec((4, 2), vec![0.2; 8]).unwrap();
        let adapter = LoraAdapter::new(lora_a, lora_b, 1.0, "test").unwrap();

        let inputs = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, // Sample 1
                5.0, 6.0, 7.0, 8.0, // Sample 2
                9.0, 10.0, 11.0, 12.0, // Sample 3
            ],
        )
        .unwrap();

        let outputs = adapter.apply_batch(&inputs).unwrap();
        assert_eq!(outputs.nrows(), 3);
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_tmp_dir() -> std::path::PathBuf {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let tmp = std::env::temp_dir().join(format!("kizzasi_lora_test_{}", ts));
        std::fs::create_dir_all(&tmp).unwrap();
        tmp
    }

    // ── round-trip tests ──────────────────────────────────────────────────────

    #[test]
    fn test_lora_save_load_roundtrip() {
        let tmp = make_tmp_dir();

        // Build an adapter with known non-zero weights: A is (4×8), B is (8×4), rank=4
        let a_data: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..32).map(|i| -(i as f32) * 0.02).collect();
        let lora_a_orig = Array2::from_shape_vec((4, 8), a_data).unwrap();
        let lora_b_orig = Array2::from_shape_vec((8, 4), b_data).unwrap();

        let config = LoraConfig::new().rank(4).alpha(8.0);
        let scaling = config.scaling();
        let adapter_orig =
            LoraAdapter::new(lora_a_orig.clone(), lora_b_orig.clone(), scaling, "rt_test").unwrap();

        let loader = LoraAdapterLoader::new(&tmp);
        loader.save("rt_test", &adapter_orig, &config).unwrap();

        let (adapter_loaded, _cfg) = loader.load("rt_test").unwrap();

        assert_eq!(adapter_loaded.rank(), adapter_orig.rank());
        assert_eq!(adapter_loaded.in_features(), adapter_orig.in_features());
        assert_eq!(adapter_loaded.out_features(), adapter_orig.out_features());

        // Verify apply() produces identical output on a fixed input
        let input = Array1::from_vec(vec![1.0, 0.5, -0.5, 0.25, 0.1, -0.1, 0.75, -0.25]);
        let out_orig = adapter_orig.apply(&input).unwrap();
        let out_loaded = adapter_loaded.apply(&input).unwrap();

        assert_eq!(out_orig.len(), out_loaded.len());
        for (a, b) in out_orig.iter().zip(out_loaded.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "apply() output differs: {} vs {}",
                a,
                b
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_lora_save_load_matrix_equality() {
        let tmp = make_tmp_dir();

        let a_data: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.05).collect();
        let b_data: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * -0.03).collect();
        let lora_a_orig = Array2::from_shape_vec((4, 8), a_data).unwrap();
        let lora_b_orig = Array2::from_shape_vec((8, 4), b_data).unwrap();

        let config = LoraConfig::new().rank(4).alpha(16.0);
        let scaling = config.scaling();
        let adapter_orig =
            LoraAdapter::new(lora_a_orig.clone(), lora_b_orig.clone(), scaling, "me_test").unwrap();

        let loader = LoraAdapterLoader::new(&tmp);
        loader.save("me_test", &adapter_orig, &config).unwrap();
        let (adapter_loaded, _cfg) = loader.load("me_test").unwrap();

        // Verify matrix A element-wise via apply() on basis vectors
        // Each basis vector e_i isolates the i-th column of A (and through B the full product).
        // A direct element comparison via the public fields (which are pub) is simpler:
        for r in 0..adapter_orig.lora_a.nrows() {
            for c in 0..adapter_orig.lora_a.ncols() {
                let orig = adapter_orig.lora_a[[r, c]];
                let loaded = adapter_loaded.lora_a[[r, c]];
                assert!(
                    (orig - loaded).abs() < 1e-6,
                    "lora_a[{},{}] differs: {} vs {}",
                    r,
                    c,
                    orig,
                    loaded
                );
            }
        }
        for r in 0..adapter_orig.lora_b.nrows() {
            for c in 0..adapter_orig.lora_b.ncols() {
                let orig = adapter_orig.lora_b[[r, c]];
                let loaded = adapter_loaded.lora_b[[r, c]];
                assert!(
                    (orig - loaded).abs() < 1e-6,
                    "lora_b[{},{}] differs: {} vs {}",
                    r,
                    c,
                    orig,
                    loaded
                );
            }
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_lora_load_missing_adapter_returns_err() {
        let tmp = make_tmp_dir();

        let loader = LoraAdapterLoader::new(&tmp);
        let result = loader.load("nonexistent_adapter_99");

        assert!(result.is_err(), "Expected Err for missing adapter, got Ok");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_lora_config_preserved_after_roundtrip() {
        let tmp = make_tmp_dir();

        let config = LoraConfig::new()
            .rank(8)
            .alpha(32.0)
            .dropout(0.1)
            .add_target_module("q_proj")
            .add_target_module("k_proj");

        let lora_a = Array2::from_shape_vec((8, 16), vec![0.01; 128]).unwrap();
        let lora_b = Array2::from_shape_vec((16, 8), vec![0.02; 128]).unwrap();
        let scaling = config.scaling();
        let adapter = LoraAdapter::new(lora_a, lora_b, scaling, "cfg_test").unwrap();

        let loader = LoraAdapterLoader::new(&tmp);
        loader.save("cfg_test", &adapter, &config).unwrap();

        let (_adapter_loaded, config_loaded) = loader.load("cfg_test").unwrap();

        assert_eq!(config_loaded.rank, 8);
        assert!(
            (config_loaded.alpha - 32.0).abs() < 1e-6,
            "alpha mismatch: {}",
            config_loaded.alpha
        );
        assert!(
            (config_loaded.dropout - 0.1).abs() < 1e-6,
            "dropout mismatch: {}",
            config_loaded.dropout
        );
        // target_modules order may vary in the default set; check both custom ones present
        assert!(config_loaded.target_modules.contains(&"q_proj".to_string()));
        assert!(config_loaded.target_modules.contains(&"k_proj".to_string()));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
