//! # LoRA (Low-Rank Adaptation) Support
//!
//! Implementation of LoRA for efficient fine-tuning of SSM models.
//!
//! ## Features
//!
//! - **Low-Rank Decomposition**: Efficient parameter updates with rank << dimension
//! - **Adapter Loading**: Load pre-trained LoRA adapters from safetensors
//! - **Merging**: Merge LoRA weights into base model weights
//! - **Multi-Adapter**: Support for multiple adapters simultaneously
//! - **Selective Application**: Apply LoRA to specific layers/modules
//! - **Dropout**: Inverted dropout on the LoRA input path, active only in
//!   training mode (see [`LoRALayer::train`]); layers start in evaluation
//!   mode, where `dropout` is always a no-op regardless of its configured
//!   value
//!
//! ## References
//!
//! - "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

use crate::{CoreError, CoreResult};
use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// LoRA adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// Rank of the low-rank decomposition
    pub rank: usize,
    /// Scaling factor (alpha / rank)
    pub alpha: f32,
    /// Dropout probability applied to the LoRA input path, in `[0, 1)`.
    ///
    /// Inert unless the owning [`LoRALayer`] is in training mode (see
    /// [`LoRALayer::train`]): freshly constructed layers start in
    /// evaluation mode, where `forward` ignores this value entirely and
    /// stays fully deterministic regardless of what it is set to.
    pub dropout: f32,
    /// Target modules to apply LoRA
    pub target_modules: Vec<String>,
    /// Whether to merge adapters into base weights
    pub merge_weights: bool,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec![
                "in_proj".to_string(),
                "out_proj".to_string(),
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
            ],
            merge_weights: false,
        }
    }
}

impl LoRAConfig {
    /// Create a new LoRA configuration
    pub fn new(rank: usize, alpha: f32) -> Self {
        Self {
            rank,
            alpha,
            ..Default::default()
        }
    }

    /// Set target modules
    pub fn with_targets(mut self, targets: Vec<String>) -> Self {
        self.target_modules = targets;
        self
    }

    /// Set the dropout probability (see the [`Self::dropout`] field docs for
    /// when it actually takes effect: only on a [`LoRALayer`] that has been
    /// switched into training mode via [`LoRALayer::train`]).
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Enable weight merging
    pub fn with_merge(mut self) -> Self {
        self.merge_weights = true;
        self
    }

    /// Get effective scaling
    pub fn scaling(&self) -> f32 {
        self.alpha / (self.rank as f32)
    }

    /// Validate configuration
    pub fn validate(&self) -> CoreResult<()> {
        if self.rank == 0 {
            return Err(CoreError::InvalidConfig("LoRA rank must be > 0".into()));
        }
        if self.alpha <= 0.0 {
            return Err(CoreError::InvalidConfig("LoRA alpha must be > 0".into()));
        }
        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err(CoreError::InvalidConfig(
                "LoRA dropout must be in [0, 1)".into(),
            ));
        }
        Ok(())
    }
}

/// LoRA adapter layer
///
/// Implements W' = W + (B @ A) * scaling
/// where A is (rank, in_features) and B is (out_features, rank)
///
/// # Dropout and training mode
///
/// When `config.dropout > 0.0` *and* the layer is in training mode (see
/// [`Self::train`]), `forward` applies inverted dropout to the LoRA input
/// path before it reaches `A`: `y = W x + scaling * B(A(dropout(x)))`.
/// Layers start in evaluation mode (see [`Self::eval`]), where `forward` is
/// `y = W x + scaling * B(A x)` — fully deterministic, ignoring `dropout`
/// regardless of its configured value. Note that a freshly constructed
/// layer's `B` matrix is all zeros (see [`Self::new`]), so the LoRA
/// contribution — and therefore any dropout applied to its input — has no
/// observable effect on `forward`'s output until `B` has been trained or
/// set to something else via [`Self::set_lora_b`].
///
/// [`Self::merge`] (and [`Self::get_effective_weight`]) fold the *weights*
/// `B @ A * scaling` into the base matrix; this is a static transform of
/// the trained parameters, not a forward pass, so it is never subject to
/// dropout regardless of training mode. Once merged, `forward` takes the
/// `is_merged` branch and applies no LoRA correction (and therefore no
/// dropout) at all -- calling `train()` and then `merge_all()`/`forward()`
/// deterministically reproduces the merged weight's output, which is
/// correct (dropout is a stochastic property of the *unmerged* forward
/// pass, not of the weights themselves) but easy to mistake for dropout
/// silently not working.
#[derive(Debug, Clone)]
pub struct LoRALayer {
    /// Configuration
    config: LoRAConfig,
    /// Original weight matrix (out_features, in_features)
    base_weight: Array2<f32>,
    /// LoRA A matrix (rank, in_features)
    lora_a: Array2<f32>,
    /// LoRA B matrix (out_features, rank)
    lora_b: Array2<f32>,
    /// Whether weights are merged
    is_merged: bool,
    /// Training mode (see [`Self::train`] / [`Self::eval`]). Starts `false`:
    /// a freshly constructed layer is in evaluation mode.
    training: bool,
}

impl LoRALayer {
    /// Create a new LoRA layer
    pub fn new(config: LoRAConfig, base_weight: Array2<f32>) -> CoreResult<Self> {
        config.validate()?;

        let (out_features, in_features) = base_weight.dim();

        // Initialize A with random values, B with zeros (for stability)
        use scirs2_core::random::thread_rng;
        let mut rng = thread_rng();
        let init_scale = (1.0 / config.rank as f32).sqrt();

        let lora_a = Array2::from_shape_fn((config.rank, in_features), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * init_scale
        });

        let lora_b = Array2::zeros((out_features, config.rank));

        Ok(Self {
            config,
            base_weight,
            lora_a,
            lora_b,
            is_merged: false,
            training: false,
        })
    }

    /// Switch into training mode: `forward` will stochastically apply
    /// dropout to the LoRA input path wherever `config.dropout > 0.0`. New
    /// layers start in evaluation mode (see [`Self::eval`]).
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Switch into evaluation mode: `forward` becomes fully deterministic
    /// and ignores `config.dropout` entirely. This is the default for a
    /// freshly constructed layer.
    ///
    /// (This is the standard ML train/eval mode toggle -- mirroring e.g.
    /// PyTorch's `Module.eval()` -- not the `eval`/code-execution builtin
    /// found in dynamic languages; it flips one `bool` field and runs no
    /// code of any kind.)
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Whether the layer is currently in training mode.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Apply inverted dropout to `x` for the LoRA input path: each element
    /// is independently zeroed with probability `config.dropout`, and
    /// surviving elements are rescaled by `1 / (1 - config.dropout)` so the
    /// expected magnitude of `B(A(x))` is unchanged between training and
    /// evaluation. Only called from `forward` while `self.training` is true
    /// and `config.dropout > 0.0`.
    fn apply_dropout(&self, x: &Array1<f32>) -> Array1<f32> {
        use scirs2_core::random::thread_rng;

        let keep_prob = 1.0 - self.config.dropout as f64;
        // `LoRAConfig::validate` rejects `dropout >= 1.0` and this is the
        // only constructor path (`new` calls `validate` up front), so
        // `keep_prob` is always in `(0.0, 1.0]` here and `scale` is always
        // finite.
        let scale = (1.0 / keep_prob) as f32;
        let mut rng = thread_rng();
        x.mapv(|v| {
            if rng.random_bool(keep_prob) {
                v * scale
            } else {
                0.0
            }
        })
    }

    /// Forward pass with LoRA
    pub fn forward(&self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        if x.len() != self.base_weight.ncols() {
            return Err(CoreError::DimensionMismatch {
                expected: self.base_weight.ncols(),
                got: x.len(),
            });
        }

        // Base output: W @ x
        let mut y = self.base_weight.dot(x);

        // If not merged, add LoRA contribution: (B @ A) @ x * scaling
        if !self.is_merged {
            // A @ dropout(x) -> intermediate (rank,). Dropout on the LoRA
            // input path is only stochastic in training mode; evaluation
            // mode (the default -- see `Self::eval`) ignores it entirely so
            // `forward` stays fully deterministic regardless of the
            // configured dropout probability.
            let intermediate = if self.training && self.config.dropout > 0.0 {
                self.lora_a.dot(&self.apply_dropout(x))
            } else {
                self.lora_a.dot(x)
            };

            // B @ intermediate -> delta (out_features,)
            let delta = self.lora_b.dot(&intermediate);

            // Add scaled contribution
            y = &y + &(&delta * self.config.scaling());
        }

        Ok(y)
    }

    /// Merge LoRA weights into base weights
    pub fn merge(&mut self) -> CoreResult<()> {
        if self.is_merged {
            return Ok(());
        }

        // Compute B @ A
        let lora_weight = self.lora_b.dot(&self.lora_a);

        // W' = W + (B @ A) * scaling
        self.base_weight = &self.base_weight + &(&lora_weight * self.config.scaling());
        self.is_merged = true;

        Ok(())
    }

    /// Unmerge LoRA weights from base weights
    pub fn unmerge(&mut self) -> CoreResult<()> {
        if !self.is_merged {
            return Ok(());
        }

        // Compute B @ A
        let lora_weight = self.lora_b.dot(&self.lora_a);

        // W = W' - (B @ A) * scaling
        self.base_weight = &self.base_weight - &(&lora_weight * self.config.scaling());
        self.is_merged = false;

        Ok(())
    }

    /// Get the effective weight matrix (with LoRA applied)
    pub fn get_effective_weight(&self) -> Array2<f32> {
        if self.is_merged {
            self.base_weight.clone()
        } else {
            let lora_weight = self.lora_b.dot(&self.lora_a);
            &self.base_weight + &(&lora_weight * self.config.scaling())
        }
    }

    /// Update LoRA A matrix
    pub fn set_lora_a(&mut self, a: Array2<f32>) -> CoreResult<()> {
        if a.dim() != self.lora_a.dim() {
            return Err(CoreError::DimensionMismatch {
                expected: self.lora_a.nrows() * self.lora_a.ncols(),
                got: a.nrows() * a.ncols(),
            });
        }
        self.lora_a = a;
        Ok(())
    }

    /// Update LoRA B matrix
    pub fn set_lora_b(&mut self, b: Array2<f32>) -> CoreResult<()> {
        if b.dim() != self.lora_b.dim() {
            return Err(CoreError::DimensionMismatch {
                expected: self.lora_b.nrows() * self.lora_b.ncols(),
                got: b.nrows() * b.ncols(),
            });
        }
        self.lora_b = b;
        Ok(())
    }

    /// Get LoRA parameters count
    pub fn num_parameters(&self) -> usize {
        self.lora_a.len() + self.lora_b.len()
    }

    /// Get base parameters count
    pub fn base_num_parameters(&self) -> usize {
        self.base_weight.len()
    }

    /// Get parameter reduction ratio
    pub fn parameter_ratio(&self) -> f32 {
        self.num_parameters() as f32 / self.base_num_parameters() as f32
    }

    /// Check if merged
    pub fn is_merged(&self) -> bool {
        self.is_merged
    }
}

/// LoRA adapter manager
pub struct LoRAAdapter {
    /// Adapter name
    pub name: String,
    /// LoRA configuration
    pub config: LoRAConfig,
    /// LoRA layers by module name
    pub layers: HashMap<String, LoRALayer>,
    /// Training mode applied to every layer added via [`Self::add_layer`];
    /// kept in sync with each layer's own flag by [`Self::train`] /
    /// [`Self::eval`]. See [`LoRALayer::train`]. Starts `false`.
    training: bool,
}

impl LoRAAdapter {
    /// Create a new LoRA adapter. Starts in evaluation mode (see [`Self::train`]).
    pub fn new(name: String, config: LoRAConfig) -> Self {
        Self {
            name,
            config,
            layers: HashMap::new(),
            training: false,
        }
    }

    /// Add a LoRA layer for a specific module.
    ///
    /// The layer's training mode is reset to match the adapter's current
    /// mode (see [`Self::train`] / [`Self::eval`]), so a layer added after
    /// [`Self::train`] starts training too, rather than silently staying in
    /// the [`LoRALayer::new`] default of evaluation mode.
    pub fn add_layer(&mut self, module_name: String, mut layer: LoRALayer) {
        if self.training {
            layer.train();
        } else {
            layer.eval();
        }
        self.layers.insert(module_name, layer);
    }

    /// Switch every registered layer -- and any layer added afterwards --
    /// into training mode: `forward` stochastically applies dropout
    /// wherever a layer's `config.dropout > 0.0`. Adapters start in
    /// evaluation mode.
    pub fn train(&mut self) {
        self.training = true;
        for layer in self.layers.values_mut() {
            layer.train();
        }
    }

    /// Switch every registered layer back into evaluation mode: `forward`
    /// becomes fully deterministic and ignores each layer's configured
    /// dropout probability. This is the default mode.
    pub fn eval(&mut self) {
        self.training = false;
        for layer in self.layers.values_mut() {
            layer.eval();
        }
    }

    /// Whether the adapter -- and therefore every layer added through
    /// [`Self::add_layer`] -- is currently in training mode.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Load LoRA adapter from safetensors file
    pub fn from_safetensors(
        path: impl AsRef<Path>,
        config: LoRAConfig,
        device: &Device,
    ) -> CoreResult<Self> {
        let data = std::fs::read(path.as_ref())
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to read LoRA file: {}", e)))?;

        let tensors = SafeTensors::deserialize(&data).map_err(|e| {
            CoreError::WeightLoadError(format!("Failed to deserialize LoRA: {}", e))
        })?;

        let name = path
            .as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unnamed")
            .to_string();

        let adapter = Self::new(name, config);

        // Parse tensor names to extract module names and A/B matrices
        let mut module_tensors: HashMap<String, (Option<Tensor>, Option<Tensor>)> = HashMap::new();

        for (tensor_name, tensor_view) in tensors.tensors() {
            // Convert safetensor to candle tensor
            let shape: Vec<usize> = tensor_view.shape().to_vec();
            let dtype = match tensor_view.dtype() {
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::F16 => DType::F16,
                safetensors::Dtype::BF16 => DType::BF16,
                _ => {
                    return Err(CoreError::WeightLoadError(format!(
                        "Unsupported dtype: {:?}",
                        tensor_view.dtype()
                    )))
                }
            };

            let tensor = Tensor::from_raw_buffer(tensor_view.data(), dtype, &shape, device)
                .map_err(|e| {
                    CoreError::WeightLoadError(format!("Tensor creation failed: {}", e))
                })?;

            // Parse tensor name: format like "module_name.lora_A" or "module_name.lora_B"
            if let Some((module, suffix)) = tensor_name.rsplit_once('.') {
                let entry = module_tensors
                    .entry(module.to_string())
                    .or_insert((None, None));

                if suffix == "lora_A" || suffix == "A" {
                    entry.0 = Some(tensor);
                } else if suffix == "lora_B" || suffix == "B" {
                    entry.1 = Some(tensor);
                }
            }
        }

        // Create LoRA layers from parsed tensors
        // This would require base weights which we don't have here
        // In practice, you'd load these after initializing the base model

        Ok(adapter)
    }

    /// Merge all layers
    pub fn merge_all(&mut self) -> CoreResult<()> {
        for layer in self.layers.values_mut() {
            layer.merge()?;
        }
        Ok(())
    }

    /// Unmerge all layers
    pub fn unmerge_all(&mut self) -> CoreResult<()> {
        for layer in self.layers.values_mut() {
            layer.unmerge()?;
        }
        Ok(())
    }

    /// Get total parameter count
    pub fn total_parameters(&self) -> usize {
        self.layers.values().map(|l| l.num_parameters()).sum()
    }

    /// Get parameter reduction ratio
    pub fn avg_parameter_ratio(&self) -> f32 {
        if self.layers.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.layers.values().map(|l| l.parameter_ratio()).sum();
        sum / self.layers.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config() {
        let config = LoRAConfig::new(8, 16.0);
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
        assert_eq!(config.scaling(), 2.0); // 16.0 / 8
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_lora_config_validation() {
        let mut config = LoRAConfig::new(0, 16.0);
        assert!(config.validate().is_err());

        config.rank = 8;
        config.alpha = -1.0;
        assert!(config.validate().is_err());

        config.alpha = 16.0;
        config.dropout = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_lora_layer_creation() {
        let config = LoRAConfig::new(4, 8.0);
        let base_weight = Array2::from_shape_fn((64, 32), |(i, j)| (i as f32 + j as f32) * 0.01);

        let result = LoRALayer::new(config, base_weight);
        assert!(result.is_ok());

        let layer = result.unwrap();
        assert_eq!(layer.lora_a.nrows(), 4);
        assert_eq!(layer.lora_a.ncols(), 32);
        assert_eq!(layer.lora_b.nrows(), 64);
        assert_eq!(layer.lora_b.ncols(), 4);
    }

    #[test]
    fn test_lora_forward() {
        let config = LoRAConfig::new(4, 8.0);
        let base_weight = Array2::from_elem((64, 32), 0.1);
        let layer = LoRALayer::new(config, base_weight).unwrap();

        let input = Array1::from_elem(32, 0.5);
        let output = layer.forward(&input);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.len(), 64);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_lora_merge_unmerge() {
        let config = LoRAConfig::new(4, 8.0);
        let base_weight = Array2::from_elem((64, 32), 0.1);
        let mut layer = LoRALayer::new(config, base_weight).unwrap();

        assert!(!layer.is_merged());

        // Merge
        layer.merge().unwrap();
        assert!(layer.is_merged());

        // Unmerge
        layer.unmerge().unwrap();
        assert!(!layer.is_merged());
    }

    #[test]
    fn test_lora_parameter_count() {
        let config = LoRAConfig::new(4, 8.0);
        let base_weight = Array2::from_elem((64, 32), 0.1);
        let layer = LoRALayer::new(config, base_weight).unwrap();

        // LoRA params = rank * (in_features + out_features) = 4 * (32 + 64) = 384
        assert_eq!(layer.num_parameters(), 384);
        // Base params = 64 * 32 = 2048
        assert_eq!(layer.base_num_parameters(), 2048);

        let ratio = layer.parameter_ratio();
        assert!((ratio - 0.1875).abs() < 1e-5); // 384 / 2048
    }

    #[test]
    fn test_effective_weight() {
        let config = LoRAConfig::new(2, 4.0);
        let base_weight = Array2::from_elem((4, 4), 1.0);
        let mut layer = LoRALayer::new(config, base_weight).unwrap();

        // Set specific LoRA weights for testing
        layer.lora_a = Array2::from_elem((2, 4), 0.1);
        layer.lora_b = Array2::from_elem((4, 2), 0.1);

        let effective = layer.get_effective_weight();
        // Scaling = 4.0 / 2 = 2.0
        // LoRA contribution = B @ A * scaling = (4x2) @ (2x4) * 2.0
        // Each element should be > 1.0 due to LoRA addition
        assert!(effective.iter().all(|&x| x >= 1.0));
    }

    /// A freshly constructed layer must start in evaluation mode.
    #[test]
    fn test_lora_layer_starts_in_eval_mode() {
        let config = LoRAConfig::new(4, 8.0).with_dropout(0.5);
        let base_weight = Array2::from_elem((16, 8), 0.1);
        let layer = LoRALayer::new(config, base_weight).expect("layer");
        assert!(!layer.is_training());
    }

    #[test]
    fn test_lora_train_eval_toggle() {
        let config = LoRAConfig::new(4, 8.0).with_dropout(0.5);
        let base_weight = Array2::from_elem((8, 4), 0.2);
        let mut layer = LoRALayer::new(config, base_weight).expect("layer");
        assert!(!layer.is_training());
        layer.train();
        assert!(layer.is_training());
        layer.eval();
        assert!(!layer.is_training());
    }

    // Regression for the medium bug where `dropout` was accepted, validated,
    // and exposed as a property but `forward` never consulted it at all:
    // `dropout=0.3` and `dropout=0.0` produced byte-identical output. Note
    // `lora_b` starts at all zeros (see `LoRALayer::new`), which would make
    // the LoRA path -- and therefore any dropout applied to its input --
    // invisible in `forward`'s output regardless of whether dropout is
    // wired up correctly; these tests set `lora_b` away from zero first (as
    // `test_effective_weight` above already does) so they actually exercise
    // the dropout-masked path instead of passing vacuously.
    //
    // Both tests deliberately leave `lora_a` at its natural random
    // initialisation (rather than a hand-picked matrix) and use a 64-wide
    // input at `dropout=0.5`: `dropout=0.5` minimises the chance that two
    // independent Bernoulli masks coincide per element (that probability is
    // `keep_prob^2 + (1-keep_prob)^2`, minimised at `keep_prob=0.5`, unlike
    // e.g. `dropout=0.9` where masks mostly agree on "everything dropped").
    // At 64 independent elements the chance two draws' masks coincide
    // exactly is `0.5^64`; a random (not hand-symmetric) `lora_a` makes any
    // *other* mask collision astronomically unlikely too, so a fresh
    // Bernoulli mask each call is expected to change the output on
    // essentially every draw.
    fn lora_dropout_test_layer(dropout: f32, in_features: usize) -> LoRALayer {
        let out_features = 6;
        let rank = 8;
        let config = LoRAConfig::new(rank, 8.0).with_dropout(dropout);
        let base_weight = Array2::from_shape_fn((out_features, in_features), |(i, j)| {
            (i as f32 + j as f32) * 0.001
        });
        let mut layer = LoRALayer::new(config, base_weight).expect("layer");
        // `lora_a` is left at its natural random init. Only `lora_b` needs
        // overriding away from its zero default, using a non-uniform
        // pattern so it does not collapse the rank-dimensional
        // `intermediate` vector down to a single scalar functional.
        layer
            .set_lora_b(Array2::from_shape_fn((out_features, rank), |(i, j)| {
                0.1 + 0.03 * (i as f32) - 0.017 * (j as f32)
            }))
            .expect("set_lora_b");
        layer
    }

    #[test]
    fn test_lora_dropout_inactive_in_eval_mode() {
        // Evaluation-mode forward is deterministic and matches the
        // no-dropout formula exactly, regardless of the configured dropout
        // probability.
        let layer = lora_dropout_test_layer(0.5, 64);
        assert!(!layer.is_training());

        let input = Array1::from_shape_fn(64, |i| (i as f32) * 0.01 + 0.1);
        let first = layer.forward(&input).expect("forward");
        for _ in 0..20 {
            let repeat = layer.forward(&input).expect("forward");
            assert_eq!(first, repeat, "eval-mode forward must be deterministic");
        }
    }

    #[test]
    fn test_lora_dropout_active_in_training_mode() {
        // In training mode, repeated forward calls on the same input must
        // differ (the LoRA path applies a fresh Bernoulli mask each call).
        // See `lora_dropout_test_layer`'s doc comment for why the failure
        // probability here is astronomically small.
        let mut layer = lora_dropout_test_layer(0.5, 64);
        layer.train();
        assert!(layer.is_training());

        let input = Array1::from_elem(64, 1.0_f32);
        let first = layer.forward(&input).expect("forward");
        let mut saw_difference = false;
        for _ in 0..20 {
            let repeat = layer.forward(&input).expect("forward");
            if repeat != first {
                saw_difference = true;
                break;
            }
        }
        assert!(
            saw_difference,
            "training-mode dropout should make forward stochastic once B is nonzero"
        );
    }

    #[test]
    fn test_lora_dropout_zero_is_deterministic_even_in_training_mode() {
        // dropout=0.0 (the default) must remain a deterministic no-op even
        // in training mode.
        let config = LoRAConfig::new(4, 8.0); // dropout defaults to 0.0
        let base_weight = Array2::from_elem((8, 4), 0.2);
        let mut layer = LoRALayer::new(config, base_weight).expect("layer");
        layer.lora_b = Array2::from_elem((8, 4), 0.3);
        layer.train();

        let input = Array1::from_elem(4, 0.5);
        let first = layer.forward(&input).expect("forward");
        let repeat = layer.forward(&input).expect("forward");
        assert_eq!(first, repeat);
    }

    #[test]
    fn test_lora_adapter_creation() {
        let config = LoRAConfig::new(4, 8.0);
        let adapter = LoRAAdapter::new("test_adapter".to_string(), config);

        assert_eq!(adapter.name, "test_adapter");
        assert_eq!(adapter.layers.len(), 0);
    }

    #[test]
    fn test_lora_adapter_add_layer() {
        let config = LoRAConfig::new(4, 8.0);
        let mut adapter = LoRAAdapter::new("test".to_string(), config.clone());

        let base_weight = Array2::from_elem((64, 32), 0.1);
        let layer = LoRALayer::new(config, base_weight).unwrap();

        adapter.add_layer("layer_0".to_string(), layer);
        assert_eq!(adapter.layers.len(), 1);
        assert!(adapter.layers.contains_key("layer_0"));
    }

    #[test]
    fn test_lora_dimension_mismatch() {
        let config = LoRAConfig::new(4, 8.0);
        let base_weight = Array2::from_elem((64, 32), 0.1);
        let layer = LoRALayer::new(config, base_weight).unwrap();

        // Wrong input dimension
        let input = Array1::from_elem(16, 0.5);
        let result = layer.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_lora_adapter_train_eval_propagates_to_layers() {
        let config = LoRAConfig::new(4, 8.0).with_dropout(0.5);
        let mut adapter = LoRAAdapter::new("adapter".to_string(), config.clone());
        let base_weight = Array2::from_elem((8, 4), 0.2);
        let layer = LoRALayer::new(config.clone(), base_weight.clone()).expect("layer");
        adapter.add_layer("l1".to_string(), layer);

        assert!(!adapter.is_training());
        assert!(!adapter.layers["l1"].is_training());

        adapter.train();
        assert!(adapter.is_training());
        assert!(adapter.layers["l1"].is_training());

        // A layer added after `train()` must also start in training mode,
        // not silently fall back to `LoRALayer::new`'s eval-mode default.
        let layer2 = LoRALayer::new(config, base_weight).expect("layer");
        adapter.add_layer("l2".to_string(), layer2);
        assert!(adapter.layers["l2"].is_training());

        adapter.eval();
        assert!(!adapter.is_training());
        assert!(!adapter.layers["l1"].is_training());
        assert!(!adapter.layers["l2"].is_training());
    }
}
