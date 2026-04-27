//! LoRA (Low-Rank Adaptation) for efficient fine-tuning
//!
//! Implements LoRA and QLoRA for parameter-efficient fine-tuning of neural networks.
//! LoRA adds trainable low-rank decomposition matrices A and B to frozen weight matrices,
//! so the effective weight becomes W' = W + (alpha/rank) * B @ A.
//!
//! # Key Benefits
//!
//! - **Memory Efficient**: Only rank * (in + out) parameters are trainable vs in * out
//! - **No Inference Latency**: LoRA weights can be merged into the original weights
//! - **Composable**: Multiple LoRA adapters can be swapped without reloading the base model
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi_model::lora::{LoraConfig, LoraAdapter};
//! use scirs2_core::ndarray::Array2;
//!
//! let config = LoraConfig::new(8, 16.0)
//!     .with_target_modules(vec!["q_proj".into(), "v_proj".into()]);
//!
//! let mut adapter = LoraAdapter::new(config);
//! let weight = Array2::zeros((512, 256));
//! adapter.add_layer("q_proj".into(), weight)?;
//! ```

use crate::error::{ModelError, ModelResult};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift64) — same approach as rwkv7
// ---------------------------------------------------------------------------

/// Simple xorshift64 PRNG for deterministic LoRA weight initialization.
struct SeededRng {
    state: u64,
}

impl SeededRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    /// Returns a float in [-1, 1)
    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        // Map u64 to [-1, 1)
        (self.state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
    }
}

// ---------------------------------------------------------------------------
// LoraConfig
// ---------------------------------------------------------------------------

/// Configuration for LoRA adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Rank of the low-rank decomposition (typically 4-64)
    pub rank: usize,
    /// Scaling factor (typically equal to rank or 2*rank)
    pub alpha: f32,
    /// Dropout probability for LoRA layers (0.0 = no dropout)
    pub dropout: f32,
    /// Which module names to apply LoRA to (e.g., "q_proj", "v_proj")
    pub target_modules: Vec<String>,
    /// Whether weight is stored as (fan_in, fan_out) instead of (fan_out, fan_in)
    pub fan_in_fan_out: bool,
}

impl LoraConfig {
    /// Create a new LoRA configuration with the given rank and alpha
    pub fn new(rank: usize, alpha: f32) -> Self {
        Self {
            rank,
            alpha,
            dropout: 0.0,
            target_modules: Vec::new(),
            fan_in_fan_out: false,
        }
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set target module names
    pub fn with_target_modules(mut self, modules: Vec<String>) -> Self {
        self.target_modules = modules;
        self
    }

    /// Set fan_in_fan_out flag
    pub fn with_fan_in_fan_out(mut self, fan_in_fan_out: bool) -> Self {
        self.fan_in_fan_out = fan_in_fan_out;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> ModelResult<()> {
        if self.rank == 0 {
            return Err(ModelError::invalid_config("LoRA rank must be > 0"));
        }
        if self.alpha <= 0.0 {
            return Err(ModelError::invalid_config("LoRA alpha must be > 0.0"));
        }
        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(ModelError::invalid_config(
                "LoRA dropout must be in [0.0, 1.0]",
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LoraLinear
// ---------------------------------------------------------------------------

/// A single LoRA-adapted linear layer.
///
/// Computes: output = W @ x + (alpha/rank) * B @ (A @ x)
///
/// - `lora_a` has shape (rank, in_features) — initialized with Kaiming uniform
/// - `lora_b` has shape (out_features, rank) — initialized with zeros
///
/// Because B starts at zero, the initial LoRA contribution is zero and the
/// model produces the same output as the original frozen weights.
#[derive(Debug, Clone)]
pub struct LoraLinear {
    /// Original frozen weight matrix (out_features, in_features)
    weight: Array2<f32>,
    /// LoRA A matrix (rank, in_features)
    lora_a: Array2<f32>,
    /// LoRA B matrix (out_features, rank)
    lora_b: Array2<f32>,
    /// Rank of decomposition
    rank: usize,
    /// Alpha scaling factor
    alpha: f32,
    /// Computed scaling = alpha / rank
    scaling: f32,
    /// Whether LoRA weights have been merged into W
    merged: bool,
    /// Whether LoRA adaptation is active
    enabled: bool,
}

impl LoraLinear {
    /// Create a new LoRA-adapted linear layer.
    ///
    /// The weight matrix should have shape (out_features, in_features).
    /// LoRA A is initialized with Kaiming-uniform values, B with zeros.
    pub fn new(weight: Array2<f32>, rank: usize, alpha: f32) -> ModelResult<Self> {
        if rank == 0 {
            return Err(ModelError::invalid_config("LoRA rank must be > 0"));
        }
        if alpha <= 0.0 {
            return Err(ModelError::invalid_config("LoRA alpha must be > 0.0"));
        }

        let (out_features, in_features) = weight.dim();
        if out_features == 0 || in_features == 0 {
            return Err(ModelError::invalid_config(
                "Weight matrix dimensions must be > 0",
            ));
        }
        if rank > out_features.min(in_features) {
            return Err(ModelError::invalid_config(format!(
                "LoRA rank ({}) must not exceed min(out_features, in_features) = {}",
                rank,
                out_features.min(in_features)
            )));
        }

        // Kaiming uniform initialization for A: scale = sqrt(2 / in_features)
        let kaiming_scale = (2.0 / in_features as f32).sqrt();
        let mut rng = SeededRng::new(42 + in_features as u64 + out_features as u64);
        let lora_a = Array2::from_shape_fn((rank, in_features), |_| rng.next_f32() * kaiming_scale);

        // B initialized to zero so initial LoRA contribution is zero
        let lora_b = Array2::zeros((out_features, rank));

        let scaling = alpha / rank as f32;

        Ok(Self {
            weight,
            lora_a,
            lora_b,
            rank,
            alpha,
            scaling,
            merged: false,
            enabled: true,
        })
    }

    /// Forward pass for a single input vector.
    ///
    /// Computes: output = W @ x + scaling * B @ (A @ x)
    pub fn forward(&self, input: &Array1<f32>) -> ModelResult<Array1<f32>> {
        let (out_features, in_features) = self.weight.dim();
        if input.len() != in_features {
            return Err(ModelError::dimension_mismatch(
                "LoraLinear forward input",
                in_features,
                input.len(),
            ));
        }

        // W @ x
        let mut output = Array1::zeros(out_features);
        for i in 0..out_features {
            let mut sum = 0.0_f32;
            for j in 0..in_features {
                sum += self.weight[[i, j]] * input[j];
            }
            output[i] = sum;
        }

        // Add LoRA contribution if enabled and not already merged
        if self.enabled && !self.merged {
            // A @ x  -> shape (rank,)
            let mut a_x = Array1::zeros(self.rank);
            for r in 0..self.rank {
                let mut sum = 0.0_f32;
                for j in 0..in_features {
                    sum += self.lora_a[[r, j]] * input[j];
                }
                a_x[r] = sum;
            }

            // B @ (A @ x) -> shape (out_features,)
            for i in 0..out_features {
                let mut sum = 0.0_f32;
                for r in 0..self.rank {
                    sum += self.lora_b[[i, r]] * a_x[r];
                }
                output[i] += self.scaling * sum;
            }
        }

        Ok(output)
    }

    /// Forward pass for a batch of inputs.
    ///
    /// Input shape: (batch_size, in_features)
    /// Output shape: (batch_size, out_features)
    pub fn forward_batch(&self, input: &Array2<f32>) -> ModelResult<Array2<f32>> {
        let (batch_size, input_dim) = input.dim();
        let (out_features, in_features) = self.weight.dim();

        if input_dim != in_features {
            return Err(ModelError::dimension_mismatch(
                "LoraLinear forward_batch input dim",
                in_features,
                input_dim,
            ));
        }

        // output = input @ W^T  (batch_size, out_features)
        let mut output = Array2::zeros((batch_size, out_features));
        for b in 0..batch_size {
            for i in 0..out_features {
                let mut sum = 0.0_f32;
                for j in 0..in_features {
                    sum += input[[b, j]] * self.weight[[i, j]];
                }
                output[[b, i]] = sum;
            }
        }

        // Add LoRA contribution
        if self.enabled && !self.merged {
            for b in 0..batch_size {
                // A @ x_b -> (rank,)
                let a_x: Vec<f32> = (0..self.rank)
                    .map(|r| {
                        let mut sum = 0.0_f32;
                        for j in 0..in_features {
                            sum += self.lora_a[[r, j]] * input[[b, j]];
                        }
                        sum
                    })
                    .collect();

                // B @ (A @ x_b) -> (out_features,)
                for i in 0..out_features {
                    let mut sum = 0.0_f32;
                    for (r, &ax_r) in a_x.iter().enumerate() {
                        sum += self.lora_b[[i, r]] * ax_r;
                    }
                    output[[b, i]] += self.scaling * sum;
                }
            }
        }

        Ok(output)
    }

    /// Merge LoRA weights into the original weight matrix for inference.
    ///
    /// After merging, forward passes use only the modified W with no extra computation.
    /// W = W + scaling * B @ A
    pub fn merge(&mut self) -> ModelResult<()> {
        if self.merged {
            return Err(ModelError::invalid_config(
                "LoRA weights are already merged",
            ));
        }

        let (out_features, in_features) = self.weight.dim();

        // W += scaling * B @ A
        for i in 0..out_features {
            for j in 0..in_features {
                let mut delta = 0.0_f32;
                for r in 0..self.rank {
                    delta += self.lora_b[[i, r]] * self.lora_a[[r, j]];
                }
                self.weight[[i, j]] += self.scaling * delta;
            }
        }

        self.merged = true;
        Ok(())
    }

    /// Unmerge LoRA weights from the original weight matrix.
    ///
    /// Restores W to its original values for continued training.
    /// W = W - scaling * B @ A
    pub fn unmerge(&mut self) -> ModelResult<()> {
        if !self.merged {
            return Err(ModelError::invalid_config("LoRA weights are not merged"));
        }

        let (out_features, in_features) = self.weight.dim();

        // W -= scaling * B @ A
        for i in 0..out_features {
            for j in 0..in_features {
                let mut delta = 0.0_f32;
                for r in 0..self.rank {
                    delta += self.lora_b[[i, r]] * self.lora_a[[r, j]];
                }
                self.weight[[i, j]] -= self.scaling * delta;
            }
        }

        self.merged = false;
        Ok(())
    }

    /// Number of trainable parameters: rank * (in_features + out_features)
    pub fn trainable_params(&self) -> usize {
        let (out_features, in_features) = self.weight.dim();
        self.rank * (in_features + out_features)
    }

    /// Total parameters including frozen weights
    pub fn total_params(&self) -> usize {
        let (out_features, in_features) = self.weight.dim();
        in_features * out_features + self.rank * (in_features + out_features)
    }

    /// Ratio of trainable to total parameters
    pub fn compression_ratio(&self) -> f32 {
        self.trainable_params() as f32 / self.total_params() as f32
    }

    /// Get reference to LoRA A matrix
    pub fn lora_a(&self) -> &Array2<f32> {
        &self.lora_a
    }

    /// Get reference to LoRA B matrix
    pub fn lora_b(&self) -> &Array2<f32> {
        &self.lora_b
    }

    /// Set the LoRA A matrix, validating dimensions
    pub fn set_lora_a(&mut self, a: Array2<f32>) -> ModelResult<()> {
        let (_, in_features) = self.weight.dim();
        let (a_rank, a_in) = a.dim();
        if a_rank != self.rank {
            return Err(ModelError::dimension_mismatch(
                "set_lora_a rank",
                self.rank,
                a_rank,
            ));
        }
        if a_in != in_features {
            return Err(ModelError::dimension_mismatch(
                "set_lora_a in_features",
                in_features,
                a_in,
            ));
        }
        self.lora_a = a;
        Ok(())
    }

    /// Set the LoRA B matrix, validating dimensions
    pub fn set_lora_b(&mut self, b: Array2<f32>) -> ModelResult<()> {
        let (out_features, _) = self.weight.dim();
        let (b_out, b_rank) = b.dim();
        if b_out != out_features {
            return Err(ModelError::dimension_mismatch(
                "set_lora_b out_features",
                out_features,
                b_out,
            ));
        }
        if b_rank != self.rank {
            return Err(ModelError::dimension_mismatch(
                "set_lora_b rank",
                self.rank,
                b_rank,
            ));
        }
        self.lora_b = b;
        Ok(())
    }

    /// Enable LoRA adaptation
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable LoRA adaptation (output equals original W @ x)
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Whether LoRA is currently enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Whether LoRA weights are merged into W
    pub fn is_merged(&self) -> bool {
        self.merged
    }

    /// Get the weight matrix reference
    pub fn weight(&self) -> &Array2<f32> {
        &self.weight
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the alpha
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Get the scaling factor
    pub fn scaling(&self) -> f32 {
        self.scaling
    }
}

// ---------------------------------------------------------------------------
// LoraAdapter
// ---------------------------------------------------------------------------

/// Summary statistics for a LoRA adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraAdapterSummary {
    /// Number of LoRA-adapted layers
    pub num_layers: usize,
    /// Total trainable parameters across all layers
    pub total_trainable: usize,
    /// Total original (frozen) parameters across all layers
    pub total_original: usize,
    /// Overall compression ratio (trainable / total)
    pub compression_ratio: f32,
    /// LoRA rank
    pub rank: usize,
    /// LoRA alpha
    pub alpha: f32,
}

/// Manages LoRA adaptation for a collection of layers
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// LoRA configuration
    config: LoraConfig,
    /// Named LoRA layers
    layers: Vec<(String, LoraLinear)>,
}

impl LoraAdapter {
    /// Create a new LoRA adapter with the given configuration
    pub fn new(config: LoraConfig) -> Self {
        Self {
            config,
            layers: Vec::new(),
        }
    }

    /// Add a layer to the adapter with the given name and weight matrix
    pub fn add_layer(&mut self, name: String, weight: Array2<f32>) -> ModelResult<()> {
        // Check for duplicate names
        if self.layers.iter().any(|(n, _)| n == &name) {
            return Err(ModelError::invalid_config(format!(
                "LoRA layer '{}' already exists",
                name
            )));
        }

        let layer = LoraLinear::new(weight, self.config.rank, self.config.alpha)?;
        self.layers.push((name, layer));
        Ok(())
    }

    /// Forward pass through a named layer
    pub fn forward_layer(&self, name: &str, input: &Array1<f32>) -> ModelResult<Array1<f32>> {
        let layer = self.get_layer(name).ok_or_else(|| {
            ModelError::invalid_config(format!("LoRA layer '{}' not found", name))
        })?;
        layer.forward(input)
    }

    /// Merge all LoRA weights into the original weight matrices
    pub fn merge_all(&mut self) -> ModelResult<()> {
        for (_, layer) in &mut self.layers {
            if !layer.is_merged() {
                layer.merge()?;
            }
        }
        Ok(())
    }

    /// Unmerge all LoRA weights from the original weight matrices
    pub fn unmerge_all(&mut self) -> ModelResult<()> {
        for (_, layer) in &mut self.layers {
            if layer.is_merged() {
                layer.unmerge()?;
            }
        }
        Ok(())
    }

    /// Total trainable parameters across all layers
    pub fn total_trainable_params(&self) -> usize {
        self.layers.iter().map(|(_, l)| l.trainable_params()).sum()
    }

    /// Total original (frozen) parameters across all layers
    pub fn total_original_params(&self) -> usize {
        self.layers
            .iter()
            .map(|(_, l)| {
                let (out, inp) = l.weight().dim();
                out * inp
            })
            .sum()
    }

    /// Overall compression ratio
    pub fn overall_compression_ratio(&self) -> f32 {
        let trainable = self.total_trainable_params();
        let total = self.total_original_params() + trainable;
        if total == 0 {
            return 0.0;
        }
        trainable as f32 / total as f32
    }

    /// Get names of all layers
    pub fn layer_names(&self) -> Vec<&str> {
        self.layers.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Get an immutable reference to a named layer
    pub fn get_layer(&self, name: &str) -> Option<&LoraLinear> {
        self.layers.iter().find(|(n, _)| n == name).map(|(_, l)| l)
    }

    /// Get a mutable reference to a named layer
    pub fn get_layer_mut(&mut self, name: &str) -> Option<&mut LoraLinear> {
        self.layers
            .iter_mut()
            .find(|(n, _)| n == name)
            .map(|(_, l)| l)
    }

    /// Get the adapter configuration
    pub fn config(&self) -> &LoraConfig {
        &self.config
    }

    /// Get a summary of the adapter
    pub fn summary(&self) -> LoraAdapterSummary {
        LoraAdapterSummary {
            num_layers: self.layers.len(),
            total_trainable: self.total_trainable_params(),
            total_original: self.total_original_params(),
            compression_ratio: self.overall_compression_ratio(),
            rank: self.config.rank,
            alpha: self.config.alpha,
        }
    }
}

// ---------------------------------------------------------------------------
// QLoRA (Quantized LoRA)
// ---------------------------------------------------------------------------

/// NF4 (Normal Float 4-bit) quantization values.
/// These are the 16 quantization levels optimized for normally distributed weights.
const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_04,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

/// QLoRA: Quantized LoRA for memory-efficient fine-tuning.
///
/// The base weight matrix is quantized to 4-bit NF4 format with group-wise
/// quantization, while LoRA matrices A and B remain in full fp32 precision.
/// This dramatically reduces memory usage for the frozen base weights.
#[derive(Debug, Clone)]
pub struct QLoraLinear {
    /// 4-bit quantized weight (two values packed per byte)
    quantized_weight: Vec<u8>,
    /// Per-group dequantization scale
    scale: Array1<f32>,
    /// Per-group zero point
    zero_point: Array1<f32>,
    /// Quantization group size
    group_size: usize,
    /// LoRA A matrix (rank, in_features) — full precision
    lora_a: Array2<f32>,
    /// LoRA B matrix (out_features, rank) — full precision
    lora_b: Array2<f32>,
    /// Output features dimension
    out_features: usize,
    /// Input features dimension
    in_features: usize,
    /// LoRA rank
    rank: usize,
    /// LoRA alpha
    alpha: f32,
    /// Computed scaling = alpha / rank
    scaling: f32,
}

impl QLoraLinear {
    /// Create a QLoRA layer from a full-precision weight matrix.
    ///
    /// The weight is quantized to 4-bit NF4 format with the given group size.
    /// LoRA matrices are initialized as in standard LoRA (A=Kaiming, B=zeros).
    pub fn from_weight(
        weight: Array2<f32>,
        rank: usize,
        alpha: f32,
        group_size: usize,
    ) -> ModelResult<Self> {
        if rank == 0 {
            return Err(ModelError::invalid_config("QLoRA rank must be > 0"));
        }
        if alpha <= 0.0 {
            return Err(ModelError::invalid_config("QLoRA alpha must be > 0.0"));
        }
        if group_size == 0 {
            return Err(ModelError::invalid_config("QLoRA group_size must be > 0"));
        }

        let (out_features, in_features) = weight.dim();
        if out_features == 0 || in_features == 0 {
            return Err(ModelError::invalid_config(
                "Weight matrix dimensions must be > 0",
            ));
        }
        if rank > out_features.min(in_features) {
            return Err(ModelError::invalid_config(format!(
                "QLoRA rank ({}) must not exceed min(out, in) = {}",
                rank,
                out_features.min(in_features)
            )));
        }

        // Flatten weight for quantization
        let total_elements = out_features * in_features;
        let num_groups = total_elements.div_ceil(group_size);

        let flat: Vec<f32> = weight.iter().copied().collect();

        let mut scale = Array1::zeros(num_groups);
        let mut zero_point = Array1::zeros(num_groups);
        // Two 4-bit values per byte
        let packed_len = total_elements.div_ceil(2);
        let mut quantized_weight = vec![0u8; packed_len];

        // Quantize group by group
        for g in 0..num_groups {
            let start = g * group_size;
            let end = (start + group_size).min(total_elements);
            let group = &flat[start..end];

            // Find absmax for the group
            let abs_max = group
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f32, f32::max)
                .max(1e-10);

            scale[g] = abs_max;
            zero_point[g] = 0.0; // symmetric quantization

            // Quantize each element to nearest NF4 level
            for (k, &val) in group.iter().enumerate() {
                let normalized = (val / abs_max).clamp(-1.0, 1.0);
                let quant_idx = find_nearest_nf4(normalized);
                let flat_idx = start + k;
                let byte_idx = flat_idx / 2;
                if flat_idx.is_multiple_of(2) {
                    quantized_weight[byte_idx] |= quant_idx;
                } else {
                    quantized_weight[byte_idx] |= quant_idx << 4;
                }
            }
        }

        // Initialize LoRA matrices
        let kaiming_scale = (2.0 / in_features as f32).sqrt();
        let mut rng = SeededRng::new(137 + in_features as u64 + out_features as u64);
        let lora_a = Array2::from_shape_fn((rank, in_features), |_| rng.next_f32() * kaiming_scale);
        let lora_b = Array2::zeros((out_features, rank));

        let scaling = alpha / rank as f32;

        Ok(Self {
            quantized_weight,
            scale,
            zero_point,
            group_size,
            lora_a,
            lora_b,
            out_features,
            in_features,
            rank,
            alpha,
            scaling,
        })
    }

    /// Dequantize the weight matrix back to full precision.
    ///
    /// This is an approximate reconstruction — quantization is lossy.
    pub fn dequantize_weight(&self) -> ModelResult<Array2<f32>> {
        let total_elements = self.out_features * self.in_features;
        let num_groups = total_elements.div_ceil(self.group_size);
        let mut flat = vec![0.0_f32; total_elements];

        for g in 0..num_groups {
            let start = g * self.group_size;
            let end = (start + self.group_size).min(total_elements);
            let s = self.scale[g];

            for (offset, val) in flat[start..end].iter_mut().enumerate() {
                let flat_idx = start + offset;
                let byte_idx = flat_idx / 2;
                let quant_idx = if flat_idx.is_multiple_of(2) {
                    self.quantized_weight[byte_idx] & 0x0F
                } else {
                    (self.quantized_weight[byte_idx] >> 4) & 0x0F
                };
                *val = NF4_LEVELS[quant_idx as usize] * s;
            }
        }

        Array2::from_shape_vec((self.out_features, self.in_features), flat).map_err(|e| {
            ModelError::invalid_config(format!("Failed to reshape dequantized weight: {}", e))
        })
    }

    /// Forward pass: dequantize weight, compute W @ x + scaling * B @ (A @ x)
    pub fn forward(&self, input: &Array1<f32>) -> ModelResult<Array1<f32>> {
        if input.len() != self.in_features {
            return Err(ModelError::dimension_mismatch(
                "QLoraLinear forward input",
                self.in_features,
                input.len(),
            ));
        }

        let weight = self.dequantize_weight()?;

        // W @ x
        let mut output = Array1::zeros(self.out_features);
        for i in 0..self.out_features {
            let mut sum = 0.0_f32;
            for j in 0..self.in_features {
                sum += weight[[i, j]] * input[j];
            }
            output[i] = sum;
        }

        // LoRA contribution: scaling * B @ (A @ x)
        let mut a_x = Array1::zeros(self.rank);
        for r in 0..self.rank {
            let mut sum = 0.0_f32;
            for j in 0..self.in_features {
                sum += self.lora_a[[r, j]] * input[j];
            }
            a_x[r] = sum;
        }

        for i in 0..self.out_features {
            let mut sum = 0.0_f32;
            for r in 0..self.rank {
                sum += self.lora_b[[i, r]] * a_x[r];
            }
            output[i] += self.scaling * sum;
        }

        Ok(output)
    }

    /// Memory saved compared to storing full fp32 weights, in bytes.
    pub fn memory_saved_bytes(&self) -> usize {
        let total_elements = self.out_features * self.in_features;
        let fp32_bytes = total_elements * 4; // 4 bytes per f32
        let packed_bytes = self.quantized_weight.len(); // 0.5 bytes per element
        let num_groups = total_elements.div_ceil(self.group_size);
        let scale_bytes = num_groups * 4; // scale: f32 per group
        let zero_point_bytes = num_groups * 4; // zero_point: f32 per group
        let quantized_total = packed_bytes + scale_bytes + zero_point_bytes;

        fp32_bytes.saturating_sub(quantized_total)
    }

    /// Number of trainable parameters (LoRA A and B)
    pub fn trainable_params(&self) -> usize {
        self.rank * (self.in_features + self.out_features)
    }

    /// Get the LoRA A matrix
    pub fn lora_a(&self) -> &Array2<f32> {
        &self.lora_a
    }

    /// Get the LoRA B matrix
    pub fn lora_b(&self) -> &Array2<f32> {
        &self.lora_b
    }

    /// Get the quantization group size
    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get output features
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get input features
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get the alpha scaling factor
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Get the per-group zero points
    pub fn zero_point(&self) -> &Array1<f32> {
        &self.zero_point
    }

    /// Get the per-group scales
    pub fn scale(&self) -> &Array1<f32> {
        &self.scale
    }
}

/// Find the nearest NF4 quantization level index for a normalized value in [-1, 1].
fn find_nearest_nf4(value: f32) -> u8 {
    let mut best_idx = 0u8;
    let mut best_dist = f32::MAX;
    for (i, &level) in NF4_LEVELS.iter().enumerate() {
        let dist = (value - level).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u8;
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Helper: create a simple weight matrix with known values
    fn make_weight(out: usize, inp: usize) -> Array2<f32> {
        Array2::from_shape_fn((out, inp), |(i, j)| (i * inp + j) as f32 * 0.01)
    }

    #[test]
    fn test_lora_linear_creation() -> ModelResult<()> {
        let weight = make_weight(64, 32);
        let lora = LoraLinear::new(weight.clone(), 8, 16.0)?;

        // B is zero, so forward should equal W @ x
        let input = Array1::from_vec(vec![1.0; 32]);
        let output_lora = lora.forward(&input)?;

        // Compute W @ x directly
        let mut output_plain = Array1::zeros(64);
        for i in 0..64 {
            let mut sum = 0.0_f32;
            for j in 0..32 {
                sum += weight[[i, j]] * input[j];
            }
            output_plain[i] = sum;
        }

        // Should be identical since B = 0
        for i in 0..64 {
            assert!(
                (output_lora[i] - output_plain[i]).abs() < 1e-5,
                "Mismatch at index {}: lora={}, plain={}",
                i,
                output_lora[i],
                output_plain[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_lora_linear_forward_with_nonzero_b() -> ModelResult<()> {
        let weight = make_weight(16, 8);
        let mut lora = LoraLinear::new(weight.clone(), 4, 8.0)?;

        // Set B to non-zero
        let b = Array2::from_shape_fn((16, 4), |(i, j)| (i + j) as f32 * 0.1);
        lora.set_lora_b(b)?;

        let input = Array1::from_vec(vec![1.0; 8]);
        let output_lora = lora.forward(&input)?;

        // Plain W @ x
        let mut output_plain = Array1::zeros(16);
        for i in 0..16 {
            let mut sum = 0.0_f32;
            for j in 0..8 {
                sum += weight[[i, j]] * input[j];
            }
            output_plain[i] = sum;
        }

        // Output should differ from plain since B != 0
        let mut any_diff = false;
        for i in 0..16 {
            if (output_lora[i] - output_plain[i]).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(
            any_diff,
            "LoRA output should differ from plain output when B != 0"
        );
        Ok(())
    }

    #[test]
    fn test_lora_linear_merge_unmerge() -> ModelResult<()> {
        let weight = make_weight(16, 8);
        let mut lora = LoraLinear::new(weight.clone(), 4, 8.0)?;

        // Set non-zero B
        let b = Array2::from_shape_fn((16, 4), |(i, j)| (i + j) as f32 * 0.01);
        lora.set_lora_b(b)?;

        let input = Array1::from_vec(vec![0.5; 8]);

        // Get output before merge
        let output_before = lora.forward(&input)?;

        // Merge
        lora.merge()?;
        assert!(lora.is_merged());

        // Output after merge should be the same
        let output_merged = lora.forward(&input)?;
        for i in 0..16 {
            assert!(
                (output_before[i] - output_merged[i]).abs() < 1e-4,
                "Merge changed output at {}: before={}, after={}",
                i,
                output_before[i],
                output_merged[i]
            );
        }

        // Unmerge
        lora.unmerge()?;
        assert!(!lora.is_merged());

        // Weight should be back to original
        for i in 0..16 {
            for j in 0..8 {
                assert!(
                    (lora.weight()[[i, j]] - weight[[i, j]]).abs() < 1e-4,
                    "Unmerge did not restore weight at [{}, {}]",
                    i,
                    j
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_lora_linear_trainable_params() -> ModelResult<()> {
        let weight = make_weight(64, 32);
        let lora = LoraLinear::new(weight, 8, 16.0)?;

        // trainable = rank * (in + out) = 8 * (32 + 64) = 768
        assert_eq!(lora.trainable_params(), 768);
        // total = 64*32 + 768 = 2048 + 768 = 2816
        assert_eq!(lora.total_params(), 2816);
        Ok(())
    }

    #[test]
    fn test_lora_linear_compression_ratio() -> ModelResult<()> {
        let weight = make_weight(256, 128);
        let lora = LoraLinear::new(weight, 8, 16.0)?;

        let ratio = lora.compression_ratio();
        // trainable = 8 * (128 + 256) = 3072
        // total = 256*128 + 3072 = 32768 + 3072 = 35840
        // ratio = 3072 / 35840 ≈ 0.0857
        assert!(
            ratio < 1.0,
            "Compression ratio should be < 1.0, got {}",
            ratio
        );
        assert!(
            ratio > 0.0,
            "Compression ratio should be > 0.0, got {}",
            ratio
        );

        let expected = 3072.0 / 35840.0;
        assert!(
            (ratio - expected).abs() < 1e-5,
            "Expected ratio ~{}, got {}",
            expected,
            ratio
        );
        Ok(())
    }

    #[test]
    fn test_lora_adapter_multi_layer() -> ModelResult<()> {
        let config = LoraConfig::new(4, 8.0).with_target_modules(vec![
            "q_proj".into(),
            "k_proj".into(),
            "v_proj".into(),
        ]);

        let mut adapter = LoraAdapter::new(config);
        adapter.add_layer("q_proj".into(), make_weight(32, 16))?;
        adapter.add_layer("k_proj".into(), make_weight(32, 16))?;
        adapter.add_layer("v_proj".into(), make_weight(32, 16))?;

        assert_eq!(adapter.layer_names().len(), 3);

        // Forward through each layer
        let input = Array1::from_vec(vec![1.0; 16]);
        for name in &["q_proj", "k_proj", "v_proj"] {
            let output = adapter.forward_layer(name, &input)?;
            assert_eq!(output.len(), 32);
        }

        // Forward through nonexistent layer should fail
        let result = adapter.forward_layer("nonexistent", &input);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_lora_adapter_merge_all() -> ModelResult<()> {
        let config = LoraConfig::new(4, 8.0);
        let mut adapter = LoraAdapter::new(config);

        adapter.add_layer("layer_0".into(), make_weight(16, 8))?;
        adapter.add_layer("layer_1".into(), make_weight(16, 8))?;

        // Set non-zero B on one layer
        if let Some(layer) = adapter.get_layer_mut("layer_0") {
            let b = Array2::from_shape_fn((16, 4), |(i, j)| (i + j) as f32 * 0.01);
            layer.set_lora_b(b)?;
        }

        let input = Array1::from_vec(vec![0.5; 8]);

        // Capture output before merge
        let out_before_0 = adapter.forward_layer("layer_0", &input)?;
        let out_before_1 = adapter.forward_layer("layer_1", &input)?;

        // Merge all
        adapter.merge_all()?;

        // Outputs should match
        let out_after_0 = adapter.forward_layer("layer_0", &input)?;
        let out_after_1 = adapter.forward_layer("layer_1", &input)?;

        for i in 0..16 {
            assert!(
                (out_before_0[i] - out_after_0[i]).abs() < 1e-4,
                "layer_0 merge changed output"
            );
            assert!(
                (out_before_1[i] - out_after_1[i]).abs() < 1e-4,
                "layer_1 merge changed output"
            );
        }
        Ok(())
    }

    #[test]
    fn test_lora_adapter_summary() -> ModelResult<()> {
        let config = LoraConfig::new(8, 16.0);
        let mut adapter = LoraAdapter::new(config);

        adapter.add_layer("proj_q".into(), make_weight(64, 32))?;
        adapter.add_layer("proj_v".into(), make_weight(64, 32))?;

        let summary = adapter.summary();
        assert_eq!(summary.num_layers, 2);
        assert_eq!(summary.rank, 8);
        assert!((summary.alpha - 16.0).abs() < 1e-6);
        // Each layer: trainable = 8*(32+64) = 768; two layers = 1536
        assert_eq!(summary.total_trainable, 1536);
        // Each layer original = 64*32 = 2048; two layers = 4096
        assert_eq!(summary.total_original, 4096);
        assert!(summary.compression_ratio > 0.0);
        assert!(summary.compression_ratio < 1.0);
        Ok(())
    }

    #[test]
    fn test_lora_disable_enable() -> ModelResult<()> {
        let weight = make_weight(16, 8);
        let mut lora = LoraLinear::new(weight.clone(), 4, 8.0)?;

        // Set non-zero B
        let b = Array2::from_shape_fn((16, 4), |(i, j)| (i + j) as f32 * 0.1);
        lora.set_lora_b(b)?;

        let input = Array1::from_vec(vec![1.0; 8]);

        // Compute plain W @ x
        let mut output_plain = Array1::zeros(16);
        for i in 0..16 {
            let mut sum = 0.0_f32;
            for j in 0..8 {
                sum += weight[[i, j]] * input[j];
            }
            output_plain[i] = sum;
        }

        // With LoRA enabled, output differs
        let output_enabled = lora.forward(&input)?;
        let mut any_diff = false;
        for i in 0..16 {
            if (output_enabled[i] - output_plain[i]).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "Enabled LoRA should produce different output");

        // Disable LoRA
        lora.disable();
        assert!(!lora.is_enabled());

        let output_disabled = lora.forward(&input)?;
        for i in 0..16 {
            assert!(
                (output_disabled[i] - output_plain[i]).abs() < 1e-5,
                "Disabled LoRA should produce same output as plain W"
            );
        }

        // Re-enable
        lora.enable();
        assert!(lora.is_enabled());
        let output_reenabled = lora.forward(&input)?;
        for i in 0..16 {
            assert!(
                (output_reenabled[i] - output_enabled[i]).abs() < 1e-5,
                "Re-enabled LoRA should match original enabled output"
            );
        }
        Ok(())
    }

    #[test]
    fn test_qlora_creation() -> ModelResult<()> {
        let weight = make_weight(32, 16);
        let qlora = QLoraLinear::from_weight(weight, 4, 8.0, 64)?;

        assert_eq!(qlora.out_features(), 32);
        assert_eq!(qlora.in_features(), 16);
        assert_eq!(qlora.rank(), 4);
        assert_eq!(qlora.group_size(), 64);
        assert_eq!(qlora.trainable_params(), 4 * (16 + 32));
        Ok(())
    }

    #[test]
    fn test_qlora_forward() -> ModelResult<()> {
        let weight = make_weight(16, 8);
        let qlora = QLoraLinear::from_weight(weight, 4, 8.0, 32)?;

        let input = Array1::from_vec(vec![1.0; 8]);
        let output = qlora.forward(&input)?;

        assert_eq!(output.len(), 16);
        // Output should be finite
        for &val in output.iter() {
            assert!(
                val.is_finite(),
                "QLoRA output contains non-finite value: {}",
                val
            );
        }
        Ok(())
    }

    #[test]
    fn test_qlora_memory_savings() -> ModelResult<()> {
        let weight = make_weight(256, 128);
        let qlora = QLoraLinear::from_weight(weight, 8, 16.0, 64)?;

        let saved = qlora.memory_saved_bytes();
        assert!(
            saved > 0,
            "QLoRA should save memory compared to fp32, got saved={} bytes",
            saved
        );

        // fp32 = 256*128*4 = 131072 bytes
        // quantized ≈ (256*128)/2 + groups*8 = 16384 + ~4096 = ~20480
        // saved ≈ 110592
        assert!(
            saved > 100_000,
            "Expected significant savings for 256x128 matrix, got {} bytes",
            saved
        );
        Ok(())
    }

    #[test]
    fn test_lora_config_validation() -> ModelResult<()> {
        // Valid config
        let config = LoraConfig::new(8, 16.0);
        assert!(config.validate().is_ok());

        // Invalid rank
        let bad_rank = LoraConfig::new(0, 16.0);
        assert!(bad_rank.validate().is_err());

        // Invalid alpha
        let bad_alpha = LoraConfig::new(8, -1.0);
        assert!(bad_alpha.validate().is_err());

        // Invalid dropout
        let bad_dropout = LoraConfig::new(8, 16.0).with_dropout(1.5);
        assert!(bad_dropout.validate().is_err());

        Ok(())
    }

    #[test]
    fn test_lora_batch_forward() -> ModelResult<()> {
        let weight = make_weight(16, 8);
        let lora = LoraLinear::new(weight, 4, 8.0)?;

        let batch = Array2::from_shape_fn((3, 8), |(b, j)| (b * 8 + j) as f32 * 0.1);
        let output = lora.forward_batch(&batch)?;

        assert_eq!(output.dim(), (3, 16));

        // Each row of batch output should match single forward
        for b in 0..3 {
            let single_input = Array1::from_vec(batch.row(b).to_vec());
            let single_output = lora.forward(&single_input)?;
            for i in 0..16 {
                assert!(
                    (output[[b, i]] - single_output[i]).abs() < 1e-4,
                    "Batch output[{},{}]={} != single output[{}]={}",
                    b,
                    i,
                    output[[b, i]],
                    i,
                    single_output[i]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_qlora_dequantize_roundtrip() -> ModelResult<()> {
        // With small values, NF4 quantization should approximately recover them
        let weight = Array2::from_shape_fn((8, 4), |(i, j)| {
            ((i as f32 - 4.0) * 0.2 + (j as f32 - 2.0) * 0.1).clamp(-0.9, 0.9)
        });

        let qlora = QLoraLinear::from_weight(weight.clone(), 2, 4.0, 16)?;
        let deq = qlora.dequantize_weight()?;

        assert_eq!(deq.dim(), (8, 4));

        // Quantization is lossy but should be in the right ballpark
        let mut max_err = 0.0_f32;
        for i in 0..8 {
            for j in 0..4 {
                let err = (weight[[i, j]] - deq[[i, j]]).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        // NF4 with small group sizes should have bounded error
        assert!(
            max_err < 0.5,
            "Maximum dequantization error {} is too large",
            max_err
        );
        Ok(())
    }
}
