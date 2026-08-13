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
use crate::optimizer::KizzasiAdamW;
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
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
    /// Device configuration (CPU, or Metal with the `metal` feature)
    pub device_config: DeviceConfig,
    /// Learning rate (initial for schedulers)
    pub learning_rate: f64,
    /// Batch size for loaders the trainer builds itself.
    ///
    /// Notably the validation loader derived from [`Self::validation_split`].
    /// A caller-supplied [`TimeSeriesDataLoader`] keeps its own batch size —
    /// the trainer never overrides it.
    ///
    /// [`TimeSeriesDataLoader`]: crate::dataloader::TimeSeriesDataLoader
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
    /// Fraction of the training series held out for validation, in `[0, 1)`.
    ///
    /// Honoured by [`Trainer::fit`] when no validation loader is supplied: the
    /// trailing `validation_split` fraction of the series is split off
    /// chronologically (never randomly — a shuffled split of sliding windows
    /// leaks future data). `0.0` disables the split.
    ///
    /// [`Trainer::fit`]: crate::training_loop::Trainer::fit
    pub validation_split: f32,
    /// Early stopping patience (epochs)
    pub early_stopping_patience: Option<usize>,
    /// Enable gradient checkpointing (saves memory by recomputing activations).
    ///
    /// When set, [`Trainer::train_epoch`] routes the step through
    /// [`TrainableSSM::forward_backward_checkpointed`], which keeps only the
    /// segment-boundary activations alive and recomputes each segment during
    /// the backward pass.
    ///
    /// [`Trainer::train_epoch`]: crate::training_loop::Trainer::train_epoch
    pub use_gradient_checkpointing: bool,
    /// Number of layers recomputed together (`None` = checkpoint every layer,
    /// i.e. a segment size of 1). Larger segments recompute less but keep more
    /// activations alive.
    pub checkpoint_segment_size: Option<usize>,
    /// Mixed precision training mode
    pub mixed_precision: MixedPrecision,
    /// Loss scaling factor for mixed precision (to prevent underflow).
    ///
    /// [`Trainer::train_epoch`] multiplies the loss by this factor before the
    /// backward pass and divides the resulting gradients by it again *before*
    /// clipping and the optimizer step, so [`Self::grad_clip`] keeps its
    /// configured meaning. Batches whose gradients overflow to non-finite
    /// values are skipped. Non-finite or non-positive values disable scaling.
    ///
    /// [`Trainer::train_epoch`]: crate::training_loop::Trainer::train_epoch
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
    /// Each layer is a standard **pre-norm residual block**:
    /// `x ← x + ssm_layer(layer_norm(x))`. Replacing `x` outright (no residual)
    /// degrades gradient flow with depth.
    ///
    /// Every layer scans from **its own** zero-initialised recurrent state.
    /// Sharing one state tensor across layers would make layer `L+1` start
    /// from layer `L`'s terminal state, which is not the recurrence this model
    /// documents.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, seq_len, input_dim]
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, seq_len, output_dim]
    pub fn forward(&self, input: &Tensor) -> CoreResult<Tensor> {
        let x = self.embed_input(input)?;
        let x = self.forward_layers(&x, 0, self.config.get_num_layers())?;
        self.project_output(&x)
    }

    /// Embed the input: `[batch, seq, input_dim] -> [batch, seq, hidden_dim]`
    fn embed_input(&self, input: &Tensor) -> CoreResult<Tensor> {
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
        x_flat
            .matmul(self.embedding_weight.as_tensor())
            .map_err(|e| CoreError::Generic(format!("Embedding forward failed: {}", e)))?
            .reshape((batch_size, seq_len, hidden_dim))
            .map_err(|e| CoreError::Generic(format!("Failed to reshape embedded: {}", e)))
    }

    /// Run layers `layer_range_start..layer_range_end` as pre-norm residual
    /// blocks, each starting from its own zero-initialised recurrent state.
    ///
    /// Shape in and out: `[batch, seq, hidden_dim]`.
    fn forward_layers(
        &self,
        x: &Tensor,
        layer_range_start: usize,
        layer_range_end: usize,
    ) -> CoreResult<Tensor> {
        let batch_size = x
            .dim(0)
            .map_err(|e| CoreError::Generic(format!("Failed to get batch dimension: {}", e)))?;
        let hidden_dim = self.config.get_hidden_dim();
        let state_dim = self.config.get_state_dim();

        let mut x = x.clone();
        for layer_idx in layer_range_start..layer_range_end {
            let mut h = Tensor::zeros(
                (batch_size, hidden_dim, state_dim),
                self.dtype,
                &self.device,
            )
            .map_err(|e| {
                CoreError::Generic(format!(
                    "Failed to create hidden state for layer {}: {}",
                    layer_idx, e
                ))
            })?;

            let normed = self.layer_norm(&x, layer_idx)?;
            let branch = self.ssm_layer(&normed, &mut h, layer_idx)?;
            x = x.add(&branch).map_err(|e| {
                CoreError::Generic(format!("Residual add failed at layer {}: {}", layer_idx, e))
            })?;
        }

        Ok(x)
    }

    /// Project to output dimension: `[batch, seq, hidden_dim] -> [batch, seq, output_dim]`
    fn project_output(&self, x: &Tensor) -> CoreResult<Tensor> {
        let batch_size = x
            .dim(0)
            .map_err(|e| CoreError::Generic(format!("Failed to get batch dimension: {}", e)))?;
        let seq_len = x
            .dim(1)
            .map_err(|e| CoreError::Generic(format!("Failed to get sequence dimension: {}", e)))?;
        let hidden_dim = self.config.get_hidden_dim();
        let output_dim = self.config.get_output_dim();

        x.reshape((batch_size * seq_len, hidden_dim))
            .map_err(|e| CoreError::Generic(format!("Failed to reshape for output: {}", e)))?
            .matmul(self.output_proj.as_tensor())
            .map_err(|e| CoreError::Generic(format!("Output projection failed: {}", e)))?
            .reshape((batch_size, seq_len, output_dim))
            .map_err(|e| CoreError::Generic(format!("Failed to reshape output: {}", e)))
    }

    /// Whether activation checkpointing is enabled for this model
    pub fn uses_gradient_checkpointing(&self) -> bool {
        self.training_config.use_gradient_checkpointing
    }

    /// Number of layers recomputed together during a checkpointed backward pass
    ///
    /// `TrainingConfig::checkpoint_segment_size == None` means "checkpoint at
    /// every layer", i.e. a segment size of 1 (maximum memory saving, maximum
    /// recomputation).
    pub fn checkpoint_segment_size(&self) -> usize {
        self.training_config
            .checkpoint_segment_size
            .unwrap_or(1)
            .max(1)
    }

    /// Forward + backward pass with segment-wise activation recomputation.
    ///
    /// This is the memory-for-compute trade honoured by
    /// [`TrainingConfig::use_gradient_checkpointing`]. Instead of keeping every
    /// layer's activations alive until the backward pass, the forward pass
    /// stores only the activation at each segment boundary; each segment is
    /// then recomputed on demand while its gradients are produced. Peak
    /// activation memory drops from *all* layers to *one segment*, at the cost
    /// of one extra forward pass per segment.
    ///
    /// # How the chaining works
    ///
    /// candle's [`Tensor::backward`] only seeds a scalar root, and it drops
    /// gradients for non-variable leaves. Both constraints are handled by
    /// re-entering each segment through a fresh [`Var`] boundary and
    /// backpropagating the surrogate scalar `Σ(y ⊙ ḡ)`, whose parameter
    /// gradient is exactly the vector-Jacobian product `Jᵀ ḡ` — the same value
    /// an uninterrupted backward pass would produce.
    ///
    /// # Arguments
    /// * `input` - `[batch, seq, input_dim]`
    /// * `target` - target tensor accepted by `loss_fn`
    /// * `loss_fn` - task loss over `(predictions, target)`
    /// * `loss_scale` - mixed-precision loss scale; applied before the backward
    ///   pass and removed from the returned gradients (non-finite or
    ///   non-positive values disable scaling)
    ///
    /// # Returns
    /// The **unscaled** loss value and the **unscaled** gradients.
    pub fn forward_backward_checkpointed<F>(
        &self,
        input: &Tensor,
        target: &Tensor,
        loss_fn: F,
        loss_scale: f32,
    ) -> CoreResult<(f32, GradStore)>
    where
        F: Fn(&Tensor, &Tensor) -> CoreResult<Tensor>,
    {
        let num_layers = self.config.get_num_layers();
        let segment_size = self.checkpoint_segment_size();
        let scale = if loss_scale.is_finite() && loss_scale > 0.0 {
            loss_scale
        } else {
            1.0
        };

        let mut segments: Vec<(usize, usize)> = Vec::new();
        let mut lo = 0usize;
        while lo < num_layers {
            let hi = (lo + segment_size).min(num_layers);
            segments.push((lo, hi));
            lo = hi;
        }

        // Phase 1 — forward, retaining only the segment boundary activations.
        // Detaching after each segment drops that segment's autograd graph, so
        // peak activation memory is one segment rather than the whole stack.
        let mut x = self.embed_input(input)?.detach();
        let mut boundaries: Vec<Tensor> = Vec::with_capacity(segments.len());
        for &(seg_lo, seg_hi) in segments.iter() {
            boundaries.push(x.clone());
            x = self.forward_layers(&x, seg_lo, seg_hi)?.detach();
        }

        // Phase 2 — head: output projection + loss.
        let head_var = Var::from_tensor(&x).map_err(|e| {
            CoreError::Generic(format!("Failed to create checkpoint boundary var: {}", e))
        })?;
        let predictions = self.project_output(head_var.as_tensor())?;
        let loss = loss_fn(&predictions, target)?;
        let loss_value = loss
            .to_dtype(DType::F32)
            .map_err(|e| CoreError::Generic(format!("Failed to cast loss to f32: {}", e)))?
            .to_vec0::<f32>()
            .map_err(|e| CoreError::Generic(format!("Failed to extract loss value: {}", e)))?;

        let root = if scale != 1.0 {
            loss.affine(scale as f64, 0.0)
                .map_err(|e| CoreError::Generic(format!("Loss scaling failed: {}", e)))?
        } else {
            loss
        };
        let mut head_grads = root
            .backward()
            .map_err(|e| CoreError::Generic(format!("Checkpointed head backward failed: {}", e)))?;

        let mut upstream = head_grads.remove(head_var.as_tensor()).ok_or_else(|| {
            CoreError::Generic("Checkpointed backward lost the head boundary gradient".to_string())
        })?;

        let mut total = GradStore::default();
        total.extend(head_grads).map_err(|e| {
            CoreError::Generic(format!("Failed to accumulate head gradients: {}", e))
        })?;

        // Phase 3 — recompute each segment and chain the VJP backwards.
        for (idx, &(seg_lo, seg_hi)) in segments.iter().enumerate().rev() {
            let boundary = boundaries.get(idx).ok_or_else(|| {
                CoreError::Generic(format!("Missing checkpoint boundary for segment {}", idx))
            })?;
            let seg_input = Var::from_tensor(boundary).map_err(|e| {
                CoreError::Generic(format!("Failed to create checkpoint boundary var: {}", e))
            })?;

            let y = self.forward_layers(seg_input.as_tensor(), seg_lo, seg_hi)?;
            let surrogate = y
                .mul(&upstream)
                .map_err(|e| CoreError::Generic(format!("Checkpoint VJP product failed: {}", e)))?
                .sum_all()
                .map_err(|e| CoreError::Generic(format!("Checkpoint VJP sum failed: {}", e)))?;

            let mut seg_grads = surrogate.backward().map_err(|e| {
                CoreError::Generic(format!(
                    "Checkpointed backward failed for layers {}..{}: {}",
                    seg_lo, seg_hi, e
                ))
            })?;

            upstream = seg_grads.remove(seg_input.as_tensor()).ok_or_else(|| {
                CoreError::Generic(format!(
                    "Checkpointed backward lost the boundary gradient for layers {}..{}",
                    seg_lo, seg_hi
                ))
            })?;

            total.extend(seg_grads).map_err(|e| {
                CoreError::Generic(format!("Failed to accumulate segment gradients: {}", e))
            })?;
        }

        // Phase 4 — embedding.
        let embedded = self.embed_input(input)?;
        let surrogate = embedded
            .mul(&upstream)
            .map_err(|e| CoreError::Generic(format!("Embedding VJP product failed: {}", e)))?
            .sum_all()
            .map_err(|e| CoreError::Generic(format!("Embedding VJP sum failed: {}", e)))?;
        let embed_grads = surrogate
            .backward()
            .map_err(|e| CoreError::Generic(format!("Embedding backward failed: {}", e)))?;
        total.extend(embed_grads).map_err(|e| {
            CoreError::Generic(format!("Failed to accumulate embedding gradients: {}", e))
        })?;

        if scale != 1.0 {
            self.scale_grad_store(&mut total, 1.0 / scale as f64)?;
        }

        Ok((loss_value, total))
    }

    /// Multiply every parameter gradient in `grads` by `factor`, in place
    fn scale_grad_store(&self, grads: &mut GradStore, factor: f64) -> CoreResult<()> {
        for var in self.varmap.all_vars() {
            let scaled = match grads.get(&var) {
                Some(g) => g
                    .affine(factor, 0.0)
                    .map_err(|e| CoreError::Generic(format!("grad scale failed: {}", e)))?,
                None => continue,
            };
            grads.insert(&var, scaled);
        }
        Ok(())
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

    /// SSM layer computation — S4D-style selective scan.
    ///
    /// Discretization (ZOH, Δ=1 implicit, A log-parameterized via exp):
    ///   `A_bar = exp(A)`   (element-wise; init at -0.5 gives stable discrete poles)
    ///   `B_bar = B`        (input matrix, no separate Δ)
    ///
    /// Recurrence for t = 0..seq_len:
    ///   `h[t] = A_bar ⊙ h[t-1] + B_bar ⊙ x[t]`   (broadcast over batch)
    ///   `y[t] = sum_s( h[t] ⊙ C ) + D ⊙ x[t]`    (contract state dim)
    fn ssm_layer(&self, x: &Tensor, h: &mut Tensor, layer_idx: usize) -> CoreResult<Tensor> {
        let a = self.a_matrices[layer_idx].as_tensor();
        let b = self.b_matrices[layer_idx].as_tensor();
        let c = self.c_matrices[layer_idx].as_tensor();
        let d = self.d_vectors[layer_idx].as_tensor();

        // Discretize: A_bar = exp(A), shape (hidden_dim, state_dim)
        let a_bar = a
            .exp()
            .map_err(|e| CoreError::Generic(format!("SSM exp(A) failed: {}", e)))?;

        let seq_len = x
            .dim(1)
            .map_err(|e| CoreError::Generic(format!("SSM get seq_len failed: {}", e)))?;

        // Accumulate output slices: each is (batch, hidden_dim)
        let mut ys: Vec<Tensor> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // x_t: (batch, hidden_dim)
            let x_t = x
                .narrow(1, t, 1)
                .map_err(|e| CoreError::Generic(format!("SSM narrow t={} failed: {}", t, e)))?
                .squeeze(1)
                .map_err(|e| CoreError::Generic(format!("SSM squeeze t={} failed: {}", t, e)))?;

            // b_x: (batch, hidden_dim, state_dim) = B_bar ⊙ x_t (broadcast state dim)
            let x_t_expanded = x_t
                .unsqueeze(candle_core::D::Minus1)
                .map_err(|e| CoreError::Generic(format!("SSM unsqueeze t={} failed: {}", t, e)))?;
            let bx = b
                .broadcast_mul(&x_t_expanded)
                .map_err(|e| CoreError::Generic(format!("SSM B*x t={} failed: {}", t, e)))?;

            // h: (batch, hidden_dim, state_dim) = A_bar ⊙ h + B_bar ⊙ x_t
            let new_h = a_bar
                .broadcast_mul(h)
                .map_err(|e| CoreError::Generic(format!("SSM A_bar*h t={} failed: {}", t, e)))?
                .broadcast_add(&bx)
                .map_err(|e| {
                    CoreError::Generic(format!("SSM state update t={} failed: {}", t, e))
                })?;

            *h = new_h;

            // y_t = sum_s( h ⊙ C ) + D ⊙ x_t  →  (batch, hidden_dim)
            let ch = h
                .broadcast_mul(c)
                .map_err(|e| CoreError::Generic(format!("SSM C*h t={} failed: {}", t, e)))?
                .sum(candle_core::D::Minus1)
                .map_err(|e| CoreError::Generic(format!("SSM output sum t={} failed: {}", t, e)))?;

            let dx = d
                .broadcast_mul(&x_t)
                .map_err(|e| CoreError::Generic(format!("SSM D*x t={} failed: {}", t, e)))?;

            let y_t = ch
                .broadcast_add(&dx)
                .map_err(|e| CoreError::Generic(format!("SSM y_t add t={} failed: {}", t, e)))?;

            ys.push(y_t);
        }

        // Stack along sequence dim: (batch, seq_len, hidden_dim)
        let y = Tensor::stack(&ys, 1)
            .map_err(|e| CoreError::Generic(format!("SSM stack failed: {}", e)))?;

        Ok(y)
    }

    /// Create an optimizer for this model
    ///
    /// Returns a [`KizzasiAdamW`], which reproduces `candle_nn::AdamW`'s update
    /// rule exactly but keeps its moment estimates addressable by parameter
    /// name so they can be checkpointed and restored.
    pub fn create_optimizer(&self) -> CoreResult<KizzasiAdamW> {
        let params = ParamsAdamW {
            lr: self.training_config.learning_rate,
            beta1: self.training_config.beta1,
            beta2: self.training_config.beta2,
            eps: self.training_config.eps,
            weight_decay: self.training_config.weight_decay,
        };

        KizzasiAdamW::from_varmap(&self.varmap, params)
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

    /// Silence a layer's SSM branch by zeroing its `C` and `D` parameters, so
    /// `ssm_layer` contributes exactly zero and only the residual survives.
    fn silence_layer(model: &TrainableSSM, layer_idx: usize) {
        let c = &model.c_matrices[layer_idx];
        let zeros_c = Tensor::zeros(c.shape(), c.dtype(), c.device()).unwrap();
        c.set(&zeros_c).unwrap();

        let d = &model.d_vectors[layer_idx];
        let zeros_d = Tensor::zeros(d.shape(), d.dtype(), d.device()).unwrap();
        d.set(&zeros_d).unwrap();
    }

    #[test]
    fn test_forward_has_pre_norm_residual() {
        // Regression: `forward` used to replace `x` with the layer output, so a
        // silenced layer produced an all-zero activation instead of passing the
        // input through. With the standard pre-norm residual, a zero branch
        // leaves the embedding untouched.
        let config = KizzasiConfig::new()
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(16)
            .state_dim(4)
            .num_layers(1);

        let model = TrainableSSM::new(config, TrainingConfig::default()).unwrap();
        silence_layer(&model, 0);

        let device = model.device().clone();
        let input = Tensor::new(
            &[[[0.1f32, -0.2, 0.3], [0.4, 0.5, -0.6], [0.7, -0.8, 0.9]]],
            &device,
        )
        .unwrap();

        let got = model.forward(&input).unwrap();

        // Expected: embedding -> (identity) -> output projection.
        let embedded = input
            .reshape((3usize, 3usize))
            .unwrap()
            .matmul(model.embedding_weight.as_tensor())
            .unwrap();
        let expected = embedded
            .matmul(model.output_proj.as_tensor())
            .unwrap()
            .reshape((1usize, 3usize, 3usize))
            .unwrap();

        let got_v = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let exp_v = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got_v.len(), exp_v.len());
        let max_abs = exp_v.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            max_abs > 1e-6,
            "test precondition: expectation must be non-trivial"
        );
        for (i, (g, e)) in got_v.iter().zip(exp_v.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "element {i}: residual path missing ({g} vs {e})"
            );
        }
    }

    #[test]
    fn test_forward_gives_each_layer_its_own_hidden_state() {
        // Regression: one `h` tensor was threaded through every layer, so layer
        // L+1 began its scan from layer L's terminal state. Each layer must
        // start from zeros.
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(8)
            .state_dim(4)
            .num_layers(2);

        let model = TrainableSSM::new(config, TrainingConfig::default()).unwrap();
        // Layer 0 contributes nothing to the activation but still accumulates a
        // non-zero recurrent state — exactly the state that used to leak.
        silence_layer(&model, 0);

        let device = model.device().clone();
        let input = Tensor::new(&[[[0.5f32, -0.4], [0.3, 0.2], [-0.1, 0.6]]], &device).unwrap();

        let got = model.forward(&input).unwrap();

        // Expected: embedding passes through layer 0 unchanged, then layer 1
        // runs from a *fresh zero* state.
        let embedded = input
            .reshape((3usize, 2usize))
            .unwrap()
            .matmul(model.embedding_weight.as_tensor())
            .unwrap()
            .reshape((1usize, 3usize, 8usize))
            .unwrap();

        let mut h = Tensor::zeros((1usize, 8usize, 4usize), model.dtype(), &device).unwrap();
        let normed = model.layer_norm(&embedded, 1).unwrap();
        let branch = model.ssm_layer(&normed, &mut h, 1).unwrap();
        let after = embedded.add(&branch).unwrap();
        let expected = after
            .reshape((3usize, 8usize))
            .unwrap()
            .matmul(model.output_proj.as_tensor())
            .unwrap()
            .reshape((1usize, 3usize, 2usize))
            .unwrap();

        let got_v = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let exp_v = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, (g, e)) in got_v.iter().zip(exp_v.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "element {i}: layer 1 did not start from a zero state ({g} vs {e})"
            );
        }

        // Sanity: the state layer 0 leaves behind is genuinely non-zero, so the
        // shared-buffer bug would have changed the result.
        let mut h0 = Tensor::zeros((1usize, 8usize, 4usize), model.dtype(), &device).unwrap();
        let normed0 = model.layer_norm(&embedded, 0).unwrap();
        let _ = model.ssm_layer(&normed0, &mut h0, 0).unwrap();
        let h0_norm = h0
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();
        assert!(
            h0_norm > 1e-6,
            "test precondition: layer 0 must leave a non-zero state, got {h0_norm}"
        );
    }

    /// Sum of squared gradients per parameter name, so two backward passes can
    /// be compared without depending on `VarMap`'s hash order.
    fn grad_snapshot(model: &TrainableSSM, grads: &GradStore) -> Vec<(String, Vec<f32>)> {
        let data = model.varmap().data().lock().expect("varmap mutex poisoned");
        let mut out: Vec<(String, Vec<f32>)> = data
            .iter()
            .map(|(name, var)| {
                let values = match grads.get(var) {
                    Some(g) => g
                        .to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                    None => Vec::new(),
                };
                (name.clone(), values)
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    fn checkpoint_test_model(
        segment_size: Option<usize>,
        layers: usize,
    ) -> (TrainableSSM, Tensor, Tensor) {
        let config = KizzasiConfig::new()
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(8)
            .state_dim(4)
            .num_layers(layers);

        let training_config = TrainingConfig::default().with_gradient_checkpointing(segment_size);

        let model = TrainableSSM::new(config, training_config).unwrap();
        let device = model.device().clone();
        let input = Tensor::new(
            &[[[0.2f32, -0.5], [0.7, 0.1], [-0.3, 0.9], [0.4, 0.4]]],
            &device,
        )
        .unwrap();
        let target = Tensor::new(
            &[[[1.0f32, 0.0], [0.0, 1.0], [-1.0, 0.5], [0.25, -0.75]]],
            &device,
        )
        .unwrap();

        (model, input, target)
    }

    fn mse(predictions: &Tensor, targets: &Tensor) -> CoreResult<Tensor> {
        predictions
            .sub(targets)
            .and_then(|d| d.sqr())
            .and_then(|d| d.mean_all())
            .map_err(|e| CoreError::Generic(format!("mse failed: {}", e)))
    }

    #[test]
    fn test_gradient_checkpointing_matches_plain_backward() {
        // Regression: `use_gradient_checkpointing` used to be a no-op. Now that
        // it recomputes activations segment by segment, the gradients it yields
        // must equal an uninterrupted backward pass to within float tolerance.
        for segment_size in [Some(1usize), Some(2), Some(4), None] {
            let (model, input, target) = checkpoint_test_model(segment_size, 4);

            let predictions = model.forward(&input).unwrap();
            let loss = mse(&predictions, &target).unwrap();
            let reference_loss = loss.to_vec0::<f32>().unwrap();
            let reference = loss.backward().unwrap();
            let reference_snapshot = grad_snapshot(&model, &reference);

            let (ckpt_loss, ckpt_grads) = model
                .forward_backward_checkpointed(&input, &target, mse, 1.0)
                .unwrap();
            let ckpt_snapshot = grad_snapshot(&model, &ckpt_grads);

            assert!(
                (ckpt_loss - reference_loss).abs() < 1e-5,
                "segment {segment_size:?}: loss mismatch {ckpt_loss} vs {reference_loss}"
            );
            assert_eq!(reference_snapshot.len(), ckpt_snapshot.len());

            let mut compared = 0usize;
            for ((r_name, r_vals), (c_name, c_vals)) in
                reference_snapshot.iter().zip(ckpt_snapshot.iter())
            {
                assert_eq!(r_name, c_name);
                assert_eq!(
                    r_vals.len(),
                    c_vals.len(),
                    "segment {segment_size:?}: {r_name} gradient presence differs"
                );
                for (i, (r, c)) in r_vals.iter().zip(c_vals.iter()).enumerate() {
                    let tol = 1e-4 * r.abs().max(1.0);
                    assert!(
                        (r - c).abs() <= tol,
                        "segment {segment_size:?}: {r_name}[{i}] = {c} but plain backward gave {r}"
                    );
                    compared += 1;
                }
            }
            assert!(
                compared > 0,
                "segment {segment_size:?}: no gradients were compared"
            );
        }
    }

    #[test]
    fn test_gradient_checkpointing_honours_loss_scale() {
        // The scaled path must return the same *unscaled* gradients.
        let (model, input, target) = checkpoint_test_model(Some(2), 3);

        let (loss_a, grads_a) = model
            .forward_backward_checkpointed(&input, &target, mse, 1.0)
            .unwrap();
        let (loss_b, grads_b) = model
            .forward_backward_checkpointed(&input, &target, mse, 256.0)
            .unwrap();

        assert!((loss_a - loss_b).abs() < 1e-6, "{loss_a} vs {loss_b}");

        let snap_a = grad_snapshot(&model, &grads_a);
        let snap_b = grad_snapshot(&model, &grads_b);
        for ((name_a, vals_a), (name_b, vals_b)) in snap_a.iter().zip(snap_b.iter()) {
            assert_eq!(name_a, name_b);
            for (i, (a, b)) in vals_a.iter().zip(vals_b.iter()).enumerate() {
                let tol = 1e-4 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "{name_a}[{i}]: loss_scale changed the unscaled gradient ({a} vs {b})"
                );
            }
        }
    }

    #[test]
    fn test_checkpoint_segment_size_defaults_to_every_layer() {
        let config = KizzasiConfig::new().num_layers(4);

        let model = TrainableSSM::new(
            config.clone(),
            TrainingConfig::default().with_gradient_checkpointing(None),
        )
        .unwrap();
        assert!(model.uses_gradient_checkpointing());
        assert_eq!(model.checkpoint_segment_size(), 1);

        let model = TrainableSSM::new(
            config.clone(),
            TrainingConfig::default().with_gradient_checkpointing(Some(3)),
        )
        .unwrap();
        assert_eq!(model.checkpoint_segment_size(), 3);

        // A zero segment size would produce an infinite segmentation loop.
        let model = TrainableSSM::new(
            config,
            TrainingConfig::default().with_gradient_checkpointing(Some(0)),
        )
        .unwrap();
        assert_eq!(model.checkpoint_segment_size(), 1);
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
