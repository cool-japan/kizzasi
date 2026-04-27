//! Training core — SchedulerType, MixedPrecision, TrainingConfig, TrainableSSM
//!
//! This module contains the foundational training infrastructure:
//!
//! - [`SchedulerType`] — learning rate scheduler variants
//! - [`MixedPrecision`] — FP16/BF16 mixed precision modes
//! - [`TrainingConfig`] — full training hyperparameter configuration
//! - [`TrainableSSM`] — differentiable SSM model with candle Var parameters

use crate::config::KizzasiConfig;
use crate::device::DeviceConfig;
use crate::error::{CoreError, CoreResult};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

/// Scheduler type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerType {
    Constant,
    Linear {
        warmup_steps: usize,
        final_lr: f64,
    },
    Cosine {
        warmup_steps: usize,
        min_lr: f64,
    },
    Step {
        milestones: Vec<usize>,
        decay_factor: f64,
    },
    Exponential {
        decay_rate: f64,
        decay_steps: usize,
    },
    OneCycle {
        warmup_pct: f64,
    },
    Polynomial {
        final_lr: f64,
        power: f64,
    },
}

/// Mixed precision training mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MixedPrecision {
    /// Full precision (FP32)
    None,
    /// Half precision (FP16) - faster but less stable
    FP16,
    /// Brain float 16 (BF16) - better stability than FP16
    BF16,
}

impl MixedPrecision {
    /// Convert to candle DType
    pub fn to_dtype(&self) -> DType {
        match self {
            MixedPrecision::None => DType::F32,
            MixedPrecision::FP16 => DType::F16,
            MixedPrecision::BF16 => DType::BF16,
        }
    }

    /// Check if mixed precision is enabled
    pub fn is_enabled(&self) -> bool {
        !matches!(self, MixedPrecision::None)
    }
}

/// Configuration for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Device configuration (CPU/CUDA/Metal)
    pub device_config: DeviceConfig,
    /// Learning rate (initial for schedulers)
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Gradient clipping threshold
    pub grad_clip: Option<f32>,
    /// Beta1 for Adam optimizer
    pub beta1: f64,
    /// Beta2 for Adam optimizer
    pub beta2: f64,
    /// Epsilon for Adam optimizer
    pub eps: f64,
    /// Learning rate scheduler type
    pub scheduler: Option<SchedulerType>,
    /// Enable metrics tracking
    pub track_metrics: bool,
    /// Log interval (batches)
    pub log_interval: usize,
    /// Validation split (0.0 to 1.0)
    pub validation_split: f32,
    /// Early stopping patience (epochs)
    pub early_stopping_patience: Option<usize>,
    /// Enable gradient checkpointing (saves memory by recomputing activations)
    pub use_gradient_checkpointing: bool,
    /// Checkpoint every N layers (None = checkpoint all layers)
    pub checkpoint_segment_size: Option<usize>,
    /// Mixed precision training mode
    pub mixed_precision: MixedPrecision,
    /// Loss scaling factor for mixed precision (to prevent underflow)
    pub loss_scale: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            device_config: DeviceConfig::default(),
            learning_rate: 1e-4,
            batch_size: 32,
            epochs: 10,
            weight_decay: 1e-2,
            grad_clip: Some(1.0),
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            scheduler: None,
            track_metrics: true,
            log_interval: 10,
            validation_split: 0.2,
            early_stopping_patience: Some(5),
            use_gradient_checkpointing: false,
            checkpoint_segment_size: Some(2), // Checkpoint every 2 layers by default
            mixed_precision: MixedPrecision::None,
            loss_scale: 1.0, // No scaling by default
        }
    }
}

impl TrainingConfig {
    /// Set scheduler type
    pub fn with_scheduler(mut self, scheduler: SchedulerType) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Disable metrics tracking
    pub fn without_metrics(mut self) -> Self {
        self.track_metrics = false;
        self
    }

    /// Set validation split
    pub fn with_validation_split(mut self, split: f32) -> Self {
        self.validation_split = split;
        self
    }

    /// Set early stopping patience
    pub fn with_early_stopping(mut self, patience: usize) -> Self {
        self.early_stopping_patience = Some(patience);
        self
    }

    /// Disable early stopping
    pub fn without_early_stopping(mut self) -> Self {
        self.early_stopping_patience = None;
        self
    }

    /// Enable gradient checkpointing for memory-efficient training
    pub fn with_gradient_checkpointing(mut self, segment_size: Option<usize>) -> Self {
        self.use_gradient_checkpointing = true;
        self.checkpoint_segment_size = segment_size;
        self
    }

    /// Disable gradient checkpointing
    pub fn without_gradient_checkpointing(mut self) -> Self {
        self.use_gradient_checkpointing = false;
        self
    }

    /// Enable mixed precision training (FP16)
    pub fn with_fp16(mut self) -> Self {
        self.mixed_precision = MixedPrecision::FP16;
        self.loss_scale = 128.0; // Default loss scale for FP16
        self
    }

    /// Enable mixed precision training (BF16)
    pub fn with_bf16(mut self) -> Self {
        self.mixed_precision = MixedPrecision::BF16;
        self.loss_scale = 1.0; // BF16 is more stable, doesn't need scaling
        self
    }

    /// Set mixed precision mode
    pub fn with_mixed_precision(mut self, mode: MixedPrecision, loss_scale: f32) -> Self {
        self.mixed_precision = mode;
        self.loss_scale = loss_scale;
        self
    }

    /// Disable mixed precision training
    pub fn without_mixed_precision(mut self) -> Self {
        self.mixed_precision = MixedPrecision::None;
        self.loss_scale = 1.0;
        self
    }
}

/// Trainable Selective SSM using candle Tensors
pub struct TrainableSSM {
    pub(crate) config: KizzasiConfig,
    pub(crate) training_config: TrainingConfig,
    pub(crate) device: Device,
    pub(crate) dtype: DType,
    // Learnable parameters
    pub(crate) embedding_weight: Var,
    pub(crate) a_matrices: Vec<Var>,
    pub(crate) b_matrices: Vec<Var>,
    pub(crate) c_matrices: Vec<Var>,
    pub(crate) d_vectors: Vec<Var>,
    pub(crate) output_proj: Var,
    // Layer normalization parameters
    pub(crate) ln_gamma: Vec<Var>,
    pub(crate) ln_beta: Vec<Var>,
    // Variable map for optimizer
    pub(crate) varmap: VarMap,
}

impl TrainableSSM {
    /// Create a new trainable SSM model
    pub fn new(config: KizzasiConfig, training_config: TrainingConfig) -> CoreResult<Self> {
        // Create device from configuration
        let device = training_config.device_config.create_device()?;

        // Use mixed precision dtype from training config
        let dtype = training_config.mixed_precision.to_dtype();

        let hidden_dim = config.get_hidden_dim();
        let state_dim = config.get_state_dim();
        let num_layers = config.get_num_layers();
        let input_dim = config.get_input_dim();
        let output_dim = config.get_output_dim();

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        // Initialize embedding layer
        let embedding_weight_tensor = vb
            .get_with_hints(
                (input_dim, hidden_dim),
                "embedding.weight",
                candle_nn::init::DEFAULT_KAIMING_NORMAL,
            )
            .map_err(|e| CoreError::Generic(format!("Failed to create embedding: {}", e)))?;
        let embedding_weight = Var::from_tensor(&embedding_weight_tensor)
            .map_err(|e| CoreError::Generic(format!("Failed to create embedding var: {}", e)))?;

        // Initialize SSM matrices for each layer
        let mut a_matrices = Vec::with_capacity(num_layers);
        let mut b_matrices = Vec::with_capacity(num_layers);
        let mut c_matrices = Vec::with_capacity(num_layers);
        let mut d_vectors = Vec::with_capacity(num_layers);
        let mut ln_gamma = Vec::with_capacity(num_layers);
        let mut ln_beta = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            // A matrix: state transition (initialized for stability)
            let a_tensor = vb
                .get_with_hints(
                    (hidden_dim, state_dim),
                    &format!("ssm.layer_{}.a", layer_idx),
                    candle_nn::init::Init::Const(-0.5),
                )
                .map_err(|e| CoreError::Generic(format!("Failed to create A matrix: {}", e)))?;
            let a = Var::from_tensor(&a_tensor)
                .map_err(|e| CoreError::Generic(format!("Failed to create A var: {}", e)))?;
            a_matrices.push(a);

            // B matrix: input projection to state
            let b_tensor = vb
                .get_with_hints(
                    (hidden_dim, state_dim),
                    &format!("ssm.layer_{}.b", layer_idx),
                    candle_nn::init::DEFAULT_KAIMING_NORMAL,
                )
                .map_err(|e| CoreError::Generic(format!("Failed to create B matrix: {}", e)))?;
            let b = Var::from_tensor(&b_tensor)
                .map_err(|e| CoreError::Generic(format!("Failed to create B var: {}", e)))?;
            b_matrices.push(b);

            // C matrix: state to output projection
            let c_tensor = vb
                .get_with_hints(
                    (hidden_dim, state_dim),
                    &format!("ssm.layer_{}.c", layer_idx),
                    candle_nn::init::DEFAULT_KAIMING_NORMAL,
                )
                .map_err(|e| CoreError::Generic(format!("Failed to create C matrix: {}", e)))?;
            let c = Var::from_tensor(&c_tensor)
                .map_err(|e| CoreError::Generic(format!("Failed to create C var: {}", e)))?;
            c_matrices.push(c);

            // D vector: skip connection
            let d_tensor = vb
                .get_with_hints(
                    hidden_dim,
                    &format!("ssm.layer_{}.d", layer_idx),
                    candle_nn::init::Init::Const(1.0),
                )
                .map_err(|e| CoreError::Generic(format!("Failed to create D vector: {}", e)))?;
            let d = Var::from_tensor(&d_tensor)
                .map_err(|e| CoreError::Generic(format!("Failed to create D var: {}", e)))?;
            d_vectors.push(d);

            // Layer normalization parameters
            let gamma_tensor = vb
                .get_with_hints(
                    hidden_dim,
                    &format!("ln.layer_{}.gamma", layer_idx),
                    candle_nn::init::Init::Const(1.0),
                )
                .map_err(|e| CoreError::Generic(format!("Failed to create LN gamma: {}", e)))?;
            let gamma = Var::from_tensor(&gamma_tensor)
                .map_err(|e| CoreError::Generic(format!("Failed to create LN gamma var: {}", e)))?;
            ln_gamma.push(gamma);

            let beta_tensor = vb
                .get_with_hints(
                    hidden_dim,
                    &format!("ln.layer_{}.beta", layer_idx),
                    candle_nn::init::Init::Const(0.0),
                )
                .map_err(|e| CoreError::Generic(format!("Failed to create LN beta: {}", e)))?;
            let beta = Var::from_tensor(&beta_tensor)
                .map_err(|e| CoreError::Generic(format!("Failed to create LN beta var: {}", e)))?;
            ln_beta.push(beta);
        }

        // Output projection
        let output_proj_tensor = vb
            .get_with_hints(
                (hidden_dim, output_dim),
                "output.proj",
                candle_nn::init::DEFAULT_KAIMING_NORMAL,
            )
            .map_err(|e| {
                CoreError::Generic(format!("Failed to create output projection: {}", e))
            })?;
        let output_proj = Var::from_tensor(&output_proj_tensor)
            .map_err(|e| CoreError::Generic(format!("Failed to create output proj var: {}", e)))?;

        Ok(Self {
            config,
            training_config,
            device,
            dtype,
            embedding_weight,
            a_matrices,
            b_matrices,
            c_matrices,
            d_vectors,
            output_proj,
            ln_gamma,
            ln_beta,
            varmap,
        })
    }

    /// Forward pass for training (tracks gradients)
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, seq_len, input_dim]
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, seq_len, output_dim]
    pub fn forward(&self, input: &Tensor) -> CoreResult<Tensor> {
        // Embed input: [batch, seq, input_dim] -> [batch, seq, hidden_dim]
        // Reshape input to [batch * seq, input_dim] for matmul, then reshape back
        let batch_size = input
            .dim(0)
            .map_err(|e| CoreError::Generic(format!("Failed to get batch dimension: {}", e)))?;
        let seq_len = input
            .dim(1)
            .map_err(|e| CoreError::Generic(format!("Failed to get sequence dimension: {}", e)))?;
        let input_dim = input
            .dim(2)
            .map_err(|e| CoreError::Generic(format!("Failed to get input dimension: {}", e)))?;

        let x_flat = input
            .reshape((batch_size * seq_len, input_dim))
            .map_err(|e| CoreError::Generic(format!("Failed to reshape input: {}", e)))?;

        let hidden_dim = self.config.get_hidden_dim();
        let x_embedded = x_flat
            .matmul(self.embedding_weight.as_tensor())
            .map_err(|e| CoreError::Generic(format!("Embedding forward failed: {}", e)))?;

        let x = x_embedded
            .reshape((batch_size, seq_len, hidden_dim))
            .map_err(|e| CoreError::Generic(format!("Failed to reshape embedded: {}", e)))?;

        // Initialize hidden state
        let state_dim = self.config.get_state_dim();

        let mut h = Tensor::zeros(
            (batch_size, hidden_dim, state_dim),
            self.dtype,
            &self.device,
        )
        .map_err(|e| CoreError::Generic(format!("Failed to create hidden state: {}", e)))?;

        let mut x = x;

        // Process through each layer
        for layer_idx in 0..self.config.get_num_layers() {
            x = self.layer_norm(&x, layer_idx)?;
            x = self.ssm_layer(&x, &mut h, layer_idx)?;
        }

        // Project to output dimension: [batch, seq, hidden_dim] -> [batch, seq, output_dim]
        // Reshape to [batch * seq, hidden_dim], matmul, then reshape back
        let x_flat = x
            .reshape((batch_size * seq_len, hidden_dim))
            .map_err(|e| CoreError::Generic(format!("Failed to reshape for output: {}", e)))?;

        let output_dim = self.config.get_output_dim();
        let output_flat = x_flat
            .matmul(self.output_proj.as_tensor())
            .map_err(|e| CoreError::Generic(format!("Output projection failed: {}", e)))?;

        let output = output_flat
            .reshape((batch_size, seq_len, output_dim))
            .map_err(|e| CoreError::Generic(format!("Failed to reshape output: {}", e)))?;

        Ok(output)
    }

    /// Apply layer normalization
    fn layer_norm(&self, x: &Tensor, layer_idx: usize) -> CoreResult<Tensor> {
        const EPS: f64 = 1e-5;

        // Compute mean and variance along the last dimension
        let mean = x
            .mean_keepdim(candle_core::D::Minus1)
            .map_err(|e| CoreError::Generic(format!("Layer norm mean failed: {}", e)))?;
        let x_centered = x.broadcast_sub(&mean).map_err(|e| {
            CoreError::Generic(format!("Layer norm variance computation failed: {}", e))
        })?;
        let variance = x_centered
            .sqr()
            .map_err(|e| CoreError::Generic(format!("Layer norm variance sqr failed: {}", e)))?
            .mean_keepdim(candle_core::D::Minus1)
            .map_err(|e| CoreError::Generic(format!("Layer norm variance mean failed: {}", e)))?;

        // Normalize: (x - mean) / sqrt(variance + eps)
        let std = (variance.affine(1.0, EPS))
            .map_err(|e| CoreError::Generic(format!("Layer norm variance add eps failed: {}", e)))?
            .sqrt()
            .map_err(|e| CoreError::Generic(format!("Layer norm sqrt failed: {}", e)))?;

        let normalized = x_centered
            .broadcast_div(&std)
            .map_err(|e| CoreError::Generic(format!("Layer norm division failed: {}", e)))?;

        // Apply affine transformation
        let gamma = self.ln_gamma[layer_idx].as_tensor();
        let beta = self.ln_beta[layer_idx].as_tensor();

        normalized
            .broadcast_mul(gamma)
            .map_err(|e| CoreError::Generic(format!("Layer norm gamma mul failed: {}", e)))?
            .broadcast_add(beta)
            .map_err(|e| CoreError::Generic(format!("Layer norm beta add failed: {}", e)))
    }

    /// SSM layer computation
    fn ssm_layer(&self, x: &Tensor, _h: &mut Tensor, layer_idx: usize) -> CoreResult<Tensor> {
        let _a = self.a_matrices[layer_idx].as_tensor();
        let _b = self.b_matrices[layer_idx].as_tensor();
        let _c = self.c_matrices[layer_idx].as_tensor();
        let d = self.d_vectors[layer_idx].as_tensor();

        // Simplified SSM step (full implementation would include selective scan)
        // For now, implementing a basic skip connection
        // TODO: Implement proper selective scan mechanism with state evolution

        // For training, we process the entire sequence in parallel (teacher forcing)
        // Output: y = D * x (simplified - full version uses state)
        let y = x
            .broadcast_mul(d)
            .map_err(|e| CoreError::Generic(format!("Skip connection failed: {}", e)))?;

        Ok(y)
    }

    /// Create an optimizer for this model
    pub fn create_optimizer(&self) -> CoreResult<AdamW> {
        let params = ParamsAdamW {
            lr: self.training_config.learning_rate,
            beta1: self.training_config.beta1,
            beta2: self.training_config.beta2,
            eps: self.training_config.eps,
            weight_decay: self.training_config.weight_decay,
        };

        AdamW::new(self.varmap.all_vars(), params)
            .map_err(|e| CoreError::Generic(format!("Failed to create optimizer: {}", e)))
    }

    /// Get the variable map for loading/saving weights
    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Save model weights to a safetensors file
    ///
    /// # Arguments
    /// * `path` - Path to save the safetensors file
    ///
    /// # Example
    /// ```rust,ignore
    /// model.save_weights("model.safetensors")?;
    /// ```
    pub fn save_weights<P: AsRef<std::path::Path>>(&self, path: P) -> CoreResult<()> {
        self.varmap
            .save(path)
            .map_err(|e| CoreError::Generic(format!("Failed to save weights: {}", e)))
    }

    /// Load model weights from a safetensors file
    ///
    /// # Arguments
    /// * `path` - Path to the safetensors file
    ///
    /// # Example
    /// ```rust,ignore
    /// model.load_weights("model.safetensors")?;
    /// ```
    pub fn load_weights<P: AsRef<std::path::Path>>(&mut self, path: P) -> CoreResult<()> {
        self.varmap
            .load(path)
            .map_err(|e| CoreError::Generic(format!("Failed to load weights: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn test_trainable_ssm_creation() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);

        let training_config = TrainingConfig::default();

        let model = TrainableSSM::new(config, training_config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_forward_pass() {
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2);

        let training_config = TrainingConfig::default();

        let model = TrainableSSM::new(config, training_config).unwrap();
        let device = model.device().clone();

        // Create dummy input: [batch=2, seq=10, input_dim=3]
        let input = Tensor::randn(0f32, 1.0, (2, 10, 3), &device).unwrap();

        let output = model.forward(&input);
        if let Err(e) = &output {
            panic!("Forward pass failed: {:?}", e);
        }

        let output = output.unwrap();
        assert_eq!(output.dims(), &[2, 10, 3]);
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 1e-4);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.epochs, 10);
        assert!(config.track_metrics);
        assert_eq!(config.validation_split, 0.2);
        assert_eq!(config.early_stopping_patience, Some(5));
    }

    #[test]
    fn test_training_config_with_scheduler() {
        let config = TrainingConfig::default().with_scheduler(SchedulerType::Cosine {
            warmup_steps: 100,
            min_lr: 1e-6,
        });

        assert!(config.scheduler.is_some());
        if let Some(SchedulerType::Cosine {
            warmup_steps,
            min_lr,
        }) = config.scheduler
        {
            assert_eq!(warmup_steps, 100);
            assert_eq!(min_lr, 1e-6);
        } else {
            panic!("Expected Cosine scheduler");
        }
    }

    #[test]
    fn test_training_config_builder() {
        let config = TrainingConfig::default()
            .with_validation_split(0.15)
            .with_early_stopping(10)
            .without_metrics();

        assert_eq!(config.validation_split, 0.15);
        assert_eq!(config.early_stopping_patience, Some(10));
        assert!(!config.track_metrics);
    }

    #[test]
    fn test_scheduler_type_constant() {
        let config = TrainingConfig::default().with_scheduler(SchedulerType::Constant);

        assert!(config.scheduler.is_some());
    }

    #[test]
    fn test_scheduler_type_step() {
        let config = TrainingConfig::default().with_scheduler(SchedulerType::Step {
            milestones: vec![100, 200, 300],
            decay_factor: 0.1,
        });

        if let Some(SchedulerType::Step {
            milestones,
            decay_factor,
        }) = config.scheduler
        {
            assert_eq!(milestones, vec![100, 200, 300]);
            assert_eq!(decay_factor, 0.1);
        } else {
            panic!("Expected Step scheduler");
        }
    }

    #[test]
    fn test_scheduler_type_onecycle() {
        let config =
            TrainingConfig::default().with_scheduler(SchedulerType::OneCycle { warmup_pct: 0.3 });

        if let Some(SchedulerType::OneCycle { warmup_pct }) = config.scheduler {
            assert_eq!(warmup_pct, 0.3);
        } else {
            panic!("Expected OneCycle scheduler");
        }
    }
}
