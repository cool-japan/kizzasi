//! Continuous (non-discrete) tokenization
//!
//! For AGSP, continuous tokenization is often preferred as it preserves
//! the full precision of the signal. This is essentially a learned
//! linear projection from signal space to embedding space.

use crate::error::{TokenizerError, TokenizerResult};
use crate::persistence::{ModelCheckpoint, ModelMetadata, ModelVersion};
use crate::SignalTokenizer;
use candle_core::{Device, Result as CandleResult, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Continuous tokenizer that projects signals to embedding space
#[derive(Debug, Clone)]
pub struct ContinuousTokenizer {
    /// Encoder projection (input_dim -> embed_dim)
    encoder: Array2<f32>,
    /// Decoder projection (embed_dim -> input_dim)
    decoder: Array2<f32>,
    /// Input dimension
    input_dim: usize,
    /// Embedding dimension
    embed_dim: usize,
}

impl ContinuousTokenizer {
    /// Create a new continuous tokenizer
    pub fn new(input_dim: usize, embed_dim: usize) -> Self {
        let mut rng = thread_rng();

        // Xavier initialization
        let enc_scale = (2.0 / (input_dim + embed_dim) as f32).sqrt();
        let encoder = Array2::from_shape_fn((input_dim, embed_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * enc_scale
        });

        let dec_scale = (2.0 / (embed_dim + input_dim) as f32).sqrt();
        let decoder = Array2::from_shape_fn((embed_dim, input_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * dec_scale
        });

        Self {
            encoder,
            decoder,
            input_dim,
            embed_dim,
        }
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Set encoder weights
    pub fn set_encoder(&mut self, weights: Array2<f32>) -> TokenizerResult<()> {
        if weights.shape() != [self.input_dim, self.embed_dim] {
            return Err(TokenizerError::dim_mismatch(
                self.input_dim * self.embed_dim,
                weights.len(),
                "dimension validation",
            ));
        }
        self.encoder = weights;
        Ok(())
    }

    /// Set decoder weights
    pub fn set_decoder(&mut self, weights: Array2<f32>) -> TokenizerResult<()> {
        if weights.shape() != [self.embed_dim, self.input_dim] {
            return Err(TokenizerError::dim_mismatch(
                self.embed_dim * self.input_dim,
                weights.len(),
                "dimension validation",
            ));
        }
        self.decoder = weights;
        Ok(())
    }

    /// Get encoder weights
    pub fn encoder(&self) -> &Array2<f32> {
        &self.encoder
    }

    /// Get decoder weights
    pub fn decoder(&self) -> &Array2<f32> {
        &self.decoder
    }
}

impl SignalTokenizer for ContinuousTokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if signal.len() != self.input_dim {
            return Err(TokenizerError::dim_mismatch(
                self.input_dim,
                signal.len(),
                "dimension validation",
            ));
        }
        Ok(signal.dot(&self.encoder))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if tokens.len() != self.embed_dim {
            return Err(TokenizerError::dim_mismatch(
                self.embed_dim,
                tokens.len(),
                "dimension validation",
            ));
        }
        Ok(tokens.dot(&self.decoder))
    }

    fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    fn vocab_size(&self) -> usize {
        0 // Continuous = no discrete vocabulary
    }
}

/// Training configuration for trainable tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Beta1 for AdamW
    pub beta1: f64,
    /// Beta2 for AdamW
    pub beta2: f64,
    /// Epsilon for AdamW
    pub eps: f64,
    /// Number of training epochs
    pub num_epochs: usize,
    /// Batch size
    pub batch_size: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            weight_decay: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            num_epochs: 100,
            batch_size: 32,
        }
    }
}

/// Reconstruction loss metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionMetrics {
    /// Mean Squared Error
    pub mse: f32,
    /// Mean Absolute Error
    pub mae: f32,
    /// Signal-to-Noise Ratio (dB)
    pub snr_db: f32,
    /// Root Mean Squared Error
    pub rmse: f32,
}

impl ReconstructionMetrics {
    /// Compute metrics from original and reconstructed signals
    pub fn compute(original: &Array1<f32>, reconstructed: &Array1<f32>) -> Self {
        assert_eq!(
            original.len(),
            reconstructed.len(),
            "Signal lengths must match"
        );

        let n = original.len() as f32;

        // MSE
        let mse: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(o, r)| (o - r).powi(2))
            .sum::<f32>()
            / n;

        // MAE
        let mae: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(o, r)| (o - r).abs())
            .sum::<f32>()
            / n;

        // RMSE
        let rmse = mse.sqrt();

        // SNR (Signal-to-Noise Ratio)
        let signal_power: f32 = original.iter().map(|x| x.powi(2)).sum::<f32>() / n;
        let noise_power = mse;
        let snr_db = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f32::INFINITY
        };

        Self {
            mse,
            mae,
            snr_db,
            rmse,
        }
    }

    /// Check if reconstruction quality is acceptable
    pub fn is_acceptable(&self, mse_threshold: f32, snr_threshold_db: f32) -> bool {
        self.mse < mse_threshold && self.snr_db > snr_threshold_db
    }
}

/// Trainable continuous tokenizer with gradient descent
pub struct TrainableContinuousTokenizer {
    /// Variable map for parameters
    varmap: VarMap,
    /// Encoder weights variable
    encoder_var: Var,
    /// Decoder weights variable
    decoder_var: Var,
    /// Input dimension
    input_dim: usize,
    /// Embedding dimension
    embed_dim: usize,
    /// Device (CPU or CUDA)
    device: Device,
}

impl TrainableContinuousTokenizer {
    /// Create a new trainable continuous tokenizer
    pub fn new(input_dim: usize, embed_dim: usize) -> CandleResult<Self> {
        let device = Device::Cpu;
        let varmap = VarMap::new();

        // Xavier initialization for encoder
        let enc_scale = (2.0 / (input_dim + embed_dim) as f32).sqrt();
        let encoder_init = Tensor::randn(0f32, 1.0, (input_dim, embed_dim), &device)?
            .affine(0.0, enc_scale as f64)?;
        let encoder_var = Var::from_tensor(&encoder_init)?;

        // Xavier initialization for decoder
        let dec_scale = (2.0 / (embed_dim + input_dim) as f32).sqrt();
        let decoder_init = Tensor::randn(0f32, 1.0, (embed_dim, input_dim), &device)?
            .affine(0.0, dec_scale as f64)?;
        let decoder_var = Var::from_tensor(&decoder_init)?;

        // Add variables to varmap
        varmap
            .data()
            .lock()
            .expect("VarMap lock should not be poisoned")
            .insert("encoder".to_string(), encoder_var.clone());
        varmap
            .data()
            .lock()
            .expect("VarMap lock should not be poisoned")
            .insert("decoder".to_string(), decoder_var.clone());

        Ok(Self {
            varmap,
            encoder_var,
            decoder_var,
            input_dim,
            embed_dim,
            device,
        })
    }

    /// Encode a signal to embeddings
    fn forward_encode(&self, signal: &Tensor) -> CandleResult<Tensor> {
        signal.matmul(self.encoder_var.as_tensor())
    }

    /// Decode embeddings back to signal
    fn forward_decode(&self, embeddings: &Tensor) -> CandleResult<Tensor> {
        embeddings.matmul(self.decoder_var.as_tensor())
    }

    /// Full forward pass (encode then decode)
    fn forward(&self, signal: &Tensor) -> CandleResult<Tensor> {
        let embeddings = self.forward_encode(signal)?;
        self.forward_decode(&embeddings)
    }

    /// Compute reconstruction loss (MSE)
    fn compute_loss(&self, original: &Tensor, reconstructed: &Tensor) -> CandleResult<Tensor> {
        let diff = (original - reconstructed)?;
        let squared = diff.sqr()?;
        squared.mean_all()
    }

    /// Train on a batch of signals
    pub fn train_batch(
        &self,
        signals: &[Array1<f32>],
        optimizer: &mut AdamW,
    ) -> TokenizerResult<f32> {
        // Convert signals to tensor
        let batch_data: Vec<f32> = signals.iter().flat_map(|s| s.iter().copied()).collect();
        let batch_tensor =
            Tensor::from_slice(&batch_data, (signals.len(), self.input_dim), &self.device)
                .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        // Forward pass
        let reconstructed = self
            .forward(&batch_tensor)
            .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        // Compute loss
        let loss = self
            .compute_loss(&batch_tensor, &reconstructed)
            .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        // Backward pass
        optimizer
            .backward_step(&loss)
            .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        // Return loss value
        let loss_val = loss
            .to_vec0::<f32>()
            .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        Ok(loss_val)
    }

    /// Train on a dataset
    pub fn train(
        &self,
        training_data: &[Array1<f32>],
        config: &TrainingConfig,
    ) -> TokenizerResult<Vec<f32>> {
        // Create optimizer
        let params = ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
        };
        let mut optimizer = AdamW::new(self.varmap.all_vars(), params).map_err(|e| {
            TokenizerError::InternalError(format!("Failed to create optimizer: {}", e))
        })?;

        let mut loss_history = Vec::with_capacity(config.num_epochs);

        // Training loop
        for epoch in 0..config.num_epochs {
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            // Process in batches
            for batch_start in (0..training_data.len()).step_by(config.batch_size) {
                let batch_end = (batch_start + config.batch_size).min(training_data.len());
                let batch = &training_data[batch_start..batch_end];

                let loss = self.train_batch(batch, &mut optimizer)?;
                epoch_loss += loss;
                num_batches += 1;
            }

            let avg_loss = epoch_loss / num_batches as f32;
            loss_history.push(avg_loss);

            // Log progress every 10 epochs
            if (epoch + 1) % 10 == 0 {
                tracing::debug!(
                    "Epoch {}/{}: Loss = {:.6}",
                    epoch + 1,
                    config.num_epochs,
                    avg_loss
                );
            }
        }

        Ok(loss_history)
    }

    /// Encode a signal (inference)
    pub fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if signal.len() != self.input_dim {
            return Err(TokenizerError::dim_mismatch(
                self.input_dim,
                signal.len(),
                "dimension validation",
            ));
        }

        let signal_data: Vec<f32> = signal.iter().copied().collect();
        let signal_tensor = Tensor::from_slice(&signal_data, (1, self.input_dim), &self.device)
            .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        let embeddings = self
            .forward_encode(&signal_tensor)
            .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        let result_vec = embeddings
            .to_vec2::<f32>()
            .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        Ok(Array1::from_vec(result_vec[0].clone()))
    }

    /// Decode embeddings (inference)
    pub fn decode(&self, embeddings: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if embeddings.len() != self.embed_dim {
            return Err(TokenizerError::dim_mismatch(
                self.embed_dim,
                embeddings.len(),
                "dimension validation",
            ));
        }

        let emb_data: Vec<f32> = embeddings.iter().copied().collect();
        let emb_tensor = Tensor::from_slice(&emb_data, (1, self.embed_dim), &self.device)
            .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        let reconstructed = self
            .forward_decode(&emb_tensor)
            .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        let result_vec = reconstructed
            .to_vec2::<f32>()
            .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        Ok(Array1::from_vec(result_vec[0].clone()))
    }

    /// Get encoder weights as Array2
    pub fn get_encoder_weights(&self) -> TokenizerResult<Array2<f32>> {
        let tensor = self.encoder_var.as_tensor();
        let data = tensor
            .to_vec2::<f32>()
            .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        let mut result = Array2::zeros((self.input_dim, self.embed_dim));
        for (i, row) in data.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                result[[i, j]] = val;
            }
        }

        Ok(result)
    }

    /// Get decoder weights as Array2
    pub fn get_decoder_weights(&self) -> TokenizerResult<Array2<f32>> {
        let tensor = self.decoder_var.as_tensor();
        let data = tensor
            .to_vec2::<f32>()
            .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        let mut result = Array2::zeros((self.embed_dim, self.input_dim));
        for (i, row) in data.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                result[[i, j]] = val;
            }
        }

        Ok(result)
    }

    /// Evaluate reconstruction quality on test data
    pub fn evaluate(&self, test_data: &[Array1<f32>]) -> TokenizerResult<ReconstructionMetrics> {
        let mut total_mse = 0.0;
        let mut total_mae = 0.0;
        let mut total_signal_power = 0.0;
        let mut total_noise_power = 0.0;
        let mut total_samples = 0;

        for signal in test_data {
            let embeddings = self.encode(signal)?;
            let reconstructed = self.decode(&embeddings)?;

            let metrics = ReconstructionMetrics::compute(signal, &reconstructed);
            total_mse += metrics.mse;
            total_mae += metrics.mae;

            let signal_power: f32 =
                signal.iter().map(|x| x.powi(2)).sum::<f32>() / signal.len() as f32;
            total_signal_power += signal_power;
            total_noise_power += metrics.mse;
            total_samples += 1;
        }

        let avg_mse = total_mse / total_samples as f32;
        let avg_mae = total_mae / total_samples as f32;
        let avg_rmse = avg_mse.sqrt();
        let avg_snr_db = if total_noise_power > 0.0 {
            10.0 * (total_signal_power / total_noise_power).log10()
        } else {
            f32::INFINITY
        };

        Ok(ReconstructionMetrics {
            mse: avg_mse,
            mae: avg_mae,
            snr_db: avg_snr_db,
            rmse: avg_rmse,
        })
    }

    /// Get embedding dimension
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Save model to checkpoint
    pub fn save_checkpoint<P: AsRef<Path>>(
        &self,
        path: P,
        version: &str,
        training_config: Option<TrainingConfig>,
        metrics: Option<ReconstructionMetrics>,
    ) -> TokenizerResult<()> {
        let version = ModelVersion::parse(version)?;

        let mut metadata = ModelMetadata::new(
            version,
            "TrainableContinuousTokenizer".to_string(),
            self.input_dim,
            self.embed_dim,
        );

        metadata.training_config = training_config;
        metadata.metrics = metrics;

        let mut checkpoint = ModelCheckpoint::new(metadata);

        // Add encoder and decoder weights
        let encoder_weights = self.get_encoder_weights()?;
        let decoder_weights = self.get_decoder_weights()?;

        checkpoint.add_array2("encoder".to_string(), &encoder_weights);
        checkpoint.add_array2("decoder".to_string(), &decoder_weights);

        checkpoint.save(path)
    }

    /// Load model from checkpoint
    pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> TokenizerResult<Self> {
        let checkpoint = ModelCheckpoint::load(path)?;

        // Verify model type
        if checkpoint.metadata.model_type != "TrainableContinuousTokenizer" {
            return Err(TokenizerError::InvalidConfig(format!(
                "Expected TrainableContinuousTokenizer, got {}",
                checkpoint.metadata.model_type
            )));
        }

        // Create new tokenizer with the right dimensions
        let input_dim = checkpoint.metadata.input_dim;
        let embed_dim = checkpoint.metadata.embed_dim;

        let mut tokenizer = Self::new(input_dim, embed_dim)
            .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        // Load weights
        let encoder_weights = checkpoint.get_array2("encoder")?;
        let decoder_weights = checkpoint.get_array2("decoder")?;

        // Set the weights by creating new tensors
        let encoder_tensor = Tensor::from_slice(
            encoder_weights
                .as_slice()
                .expect("Encoder weights must have contiguous layout"),
            (input_dim, embed_dim),
            &tokenizer.device,
        )
        .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        let decoder_tensor = Tensor::from_slice(
            decoder_weights
                .as_slice()
                .expect("Decoder weights must have contiguous layout"),
            (embed_dim, input_dim),
            &tokenizer.device,
        )
        .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        // Update the variables
        tokenizer.encoder_var = Var::from_tensor(&encoder_tensor)
            .map_err(|e| TokenizerError::InternalError(e.to_string()))?;
        tokenizer.decoder_var = Var::from_tensor(&decoder_tensor)
            .map_err(|e| TokenizerError::InternalError(e.to_string()))?;

        // Update varmap
        tokenizer
            .varmap
            .data()
            .lock()
            .expect("VarMap lock should not be poisoned")
            .insert("encoder".to_string(), tokenizer.encoder_var.clone());
        tokenizer
            .varmap
            .data()
            .lock()
            .expect("VarMap lock should not be poisoned")
            .insert("decoder".to_string(), tokenizer.decoder_var.clone());

        Ok(tokenizer)
    }

    /// Get metadata from a checkpoint file without loading the full model
    pub fn peek_checkpoint<P: AsRef<Path>>(path: P) -> TokenizerResult<ModelMetadata> {
        let checkpoint = ModelCheckpoint::load(path)?;
        Ok(checkpoint.metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_tokenizer() {
        let tokenizer = ContinuousTokenizer::new(3, 64);

        let signal = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let encoded = tokenizer.encode(&signal).unwrap();
        assert_eq!(encoded.len(), 64);

        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded.len(), 3);
    }

    #[test]
    fn test_dimension_mismatch() {
        let tokenizer = ContinuousTokenizer::new(3, 64);
        let signal = Array1::from_vec(vec![0.1, 0.2]); // Wrong size
        assert!(tokenizer.encode(&signal).is_err());
    }

    #[test]
    fn test_reconstruction_metrics() {
        let original = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let reconstructed = Array1::from_vec(vec![1.1, 1.9, 3.1, 3.9]);

        let metrics = ReconstructionMetrics::compute(&original, &reconstructed);

        assert!(metrics.mse > 0.0);
        assert!(metrics.mae > 0.0);
        assert!(metrics.rmse > 0.0);
        assert!(metrics.snr_db.is_finite());
        assert!(metrics.snr_db > 0.0); // Should be positive for small errors
    }

    #[test]
    fn test_reconstruction_metrics_perfect() {
        let original = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let reconstructed = original.clone();

        let metrics = ReconstructionMetrics::compute(&original, &reconstructed);

        assert_eq!(metrics.mse, 0.0);
        assert_eq!(metrics.mae, 0.0);
        assert_eq!(metrics.rmse, 0.0);
        assert!(metrics.snr_db.is_infinite());
    }

    #[test]
    fn test_metrics_is_acceptable() {
        let original = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let reconstructed = Array1::from_vec(vec![1.01, 2.01, 3.01, 4.01]);

        let metrics = ReconstructionMetrics::compute(&original, &reconstructed);

        assert!(metrics.is_acceptable(0.01, 10.0)); // Low MSE threshold, low SNR threshold
        assert!(!metrics.is_acceptable(0.0001, 100.0)); // Very strict thresholds
    }

    #[test]
    fn test_trainable_tokenizer_creation() {
        let tokenizer = TrainableContinuousTokenizer::new(8, 16).unwrap();

        assert_eq!(tokenizer.input_dim(), 8);
        assert_eq!(tokenizer.embed_dim(), 16);
    }

    #[test]
    fn test_trainable_encode_decode() {
        let tokenizer = TrainableContinuousTokenizer::new(8, 16).unwrap();

        let signal = Array1::from_vec((0..8).map(|i| i as f32 * 0.1).collect());
        let embeddings = tokenizer.encode(&signal).unwrap();
        let reconstructed = tokenizer.decode(&embeddings).unwrap();

        assert_eq!(embeddings.len(), 16);
        assert_eq!(reconstructed.len(), 8);
    }

    #[test]
    fn test_trainable_tokenizer_training() {
        let tokenizer = TrainableContinuousTokenizer::new(4, 8).unwrap();

        // Generate synthetic training data
        let training_data: Vec<Array1<f32>> = (0..50)
            .map(|i| Array1::from_vec((0..4).map(|j| ((i + j) as f32 * 0.1).sin()).collect()))
            .collect();

        let config = TrainingConfig {
            num_epochs: 10,
            batch_size: 8,
            learning_rate: 1e-3,
            ..Default::default()
        };

        let loss_history = tokenizer.train(&training_data, &config).unwrap();

        assert_eq!(loss_history.len(), 10);
        // Loss should generally decrease
        assert!(loss_history[loss_history.len() - 1] < loss_history[0] * 2.0);
    }

    #[test]
    fn test_trainable_tokenizer_evaluation() {
        let tokenizer = TrainableContinuousTokenizer::new(4, 8).unwrap();

        // Generate test data
        let test_data: Vec<Array1<f32>> = (0..10)
            .map(|i| Array1::from_vec((0..4).map(|j| ((i + j) as f32 * 0.1).cos()).collect()))
            .collect();

        let metrics = tokenizer.evaluate(&test_data).unwrap();

        assert!(metrics.mse >= 0.0);
        assert!(metrics.mae >= 0.0);
        assert!(metrics.rmse >= 0.0);
        assert!(metrics.snr_db.is_finite() || metrics.snr_db.is_infinite());
    }

    #[test]
    fn test_trainable_get_weights() {
        let tokenizer = TrainableContinuousTokenizer::new(4, 8).unwrap();

        let encoder_weights = tokenizer.get_encoder_weights().unwrap();
        let decoder_weights = tokenizer.get_decoder_weights().unwrap();

        assert_eq!(encoder_weights.shape(), &[4, 8]);
        assert_eq!(decoder_weights.shape(), &[8, 4]);
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();

        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.num_epochs, 100);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_trainable_convergence() {
        // Test that training actually improves reconstruction
        let tokenizer = TrainableContinuousTokenizer::new(8, 16).unwrap();

        // Create a simple dataset (sine waves with different frequencies)
        let training_data: Vec<Array1<f32>> = (0..100)
            .map(|i| {
                let freq = (i % 5 + 1) as f32 * 0.1;
                Array1::from_vec((0..8).map(|j| (j as f32 * freq).sin()).collect())
            })
            .collect();

        // Evaluate before training
        let metrics_before = tokenizer.evaluate(&training_data[..10]).unwrap();

        // Train
        let config = TrainingConfig {
            num_epochs: 20,
            batch_size: 16,
            learning_rate: 1e-2,
            ..Default::default()
        };
        tokenizer.train(&training_data, &config).unwrap();

        // Evaluate after training
        let metrics_after = tokenizer.evaluate(&training_data[..10]).unwrap();

        // MSE should decrease after training
        assert!(
            metrics_after.mse < metrics_before.mse,
            "MSE should decrease: before={}, after={}",
            metrics_before.mse,
            metrics_after.mse
        );

        // SNR should improve (increase) after training
        if metrics_before.snr_db.is_finite() {
            assert!(metrics_after.snr_db > metrics_before.snr_db);
        }
    }

    #[test]
    fn test_save_load_checkpoint() {
        use std::env;

        let temp_dir = env::temp_dir();
        let checkpoint_path = temp_dir.join("test_trainable_checkpoint.safetensors");

        // Create and train a tokenizer
        let tokenizer = TrainableContinuousTokenizer::new(4, 8).unwrap();

        let training_data: Vec<Array1<f32>> = (0..20)
            .map(|i| Array1::from_vec((0..4).map(|j| ((i + j) as f32 * 0.1).sin()).collect()))
            .collect();

        let config = TrainingConfig {
            num_epochs: 5,
            batch_size: 4,
            learning_rate: 1e-3,
            ..Default::default()
        };

        tokenizer.train(&training_data, &config).unwrap();

        // Evaluate before saving
        let metrics_before = tokenizer.evaluate(&training_data[..5]).unwrap();

        // Save checkpoint
        tokenizer
            .save_checkpoint(
                &checkpoint_path,
                "1.0.0",
                Some(config.clone()),
                Some(metrics_before.clone()),
            )
            .unwrap();

        // Load checkpoint
        let loaded_tokenizer =
            TrainableContinuousTokenizer::load_checkpoint(&checkpoint_path).unwrap();

        // Verify dimensions
        assert_eq!(loaded_tokenizer.input_dim(), 4);
        assert_eq!(loaded_tokenizer.embed_dim(), 8);

        // Evaluate loaded model - should have similar performance
        let metrics_loaded = loaded_tokenizer.evaluate(&training_data[..5]).unwrap();

        // Metrics should be very close
        assert!(
            (metrics_loaded.mse - metrics_before.mse).abs() < 1e-4,
            "Loaded model MSE should match: before={}, loaded={}",
            metrics_before.mse,
            metrics_loaded.mse
        );

        // Test encoding/decoding with loaded model
        let test_signal = Array1::from_vec((0..4).map(|i| (i as f32) * 0.1).collect());
        let encoded_original = tokenizer.encode(&test_signal).unwrap();
        let encoded_loaded = loaded_tokenizer.encode(&test_signal).unwrap();

        // Encodings should be identical
        for (o, l) in encoded_original.iter().zip(encoded_loaded.iter()) {
            assert!(
                (o - l).abs() < 1e-4,
                "Encoded values should match: original={}, loaded={}",
                o,
                l
            );
        }

        // Cleanup
        std::fs::remove_file(&checkpoint_path).ok();
    }

    #[test]
    fn test_peek_checkpoint() {
        use std::env;

        let temp_dir = env::temp_dir();
        let checkpoint_path = temp_dir.join("test_peek_checkpoint.safetensors");

        let tokenizer = TrainableContinuousTokenizer::new(6, 12).unwrap();

        let config = TrainingConfig {
            num_epochs: 1,
            batch_size: 4,
            ..Default::default()
        };

        tokenizer
            .save_checkpoint(&checkpoint_path, "2.1.3", Some(config.clone()), None)
            .unwrap();

        // Peek at metadata without loading full model
        let metadata = TrainableContinuousTokenizer::peek_checkpoint(&checkpoint_path).unwrap();

        assert_eq!(metadata.model_type, "TrainableContinuousTokenizer");
        assert_eq!(metadata.input_dim, 6);
        assert_eq!(metadata.embed_dim, 12);
        assert_eq!(metadata.version.to_string(), "2.1.3");
        assert!(metadata.training_config.is_some());

        // Cleanup
        std::fs::remove_file(&checkpoint_path).ok();
    }

    #[test]
    fn test_checkpoint_version_compatibility() {
        use std::env;

        let temp_dir = env::temp_dir();
        let checkpoint_path = temp_dir.join("test_version_checkpoint.safetensors");

        let tokenizer = TrainableContinuousTokenizer::new(4, 8).unwrap();

        tokenizer
            .save_checkpoint(&checkpoint_path, "1.0.0", None, None)
            .unwrap();

        // Load and check version
        let metadata = TrainableContinuousTokenizer::peek_checkpoint(&checkpoint_path).unwrap();

        let current_version = ModelVersion::new(1, 0, 0);
        assert!(metadata.version.is_compatible_with(&current_version));

        let incompatible_version = ModelVersion::new(2, 0, 0);
        assert!(!metadata.version.is_compatible_with(&incompatible_version));

        // Cleanup
        std::fs::remove_file(&checkpoint_path).ok();
    }

    #[test]
    fn test_save_checkpoint_with_metrics() {
        use std::env;

        let temp_dir = env::temp_dir();
        let checkpoint_path = temp_dir.join("test_metrics_checkpoint.safetensors");

        let tokenizer = TrainableContinuousTokenizer::new(4, 8).unwrap();

        let test_data: Vec<Array1<f32>> = (0..10)
            .map(|i| Array1::from_vec((0..4).map(|j| ((i + j) as f32 * 0.1).cos()).collect()))
            .collect();

        let metrics = tokenizer.evaluate(&test_data).unwrap();

        tokenizer
            .save_checkpoint(&checkpoint_path, "1.0.0", None, Some(metrics.clone()))
            .unwrap();

        // Load and verify metrics
        let checkpoint = crate::persistence::ModelCheckpoint::load(&checkpoint_path).unwrap();
        assert!(checkpoint.metadata.metrics.is_some());

        let loaded_metrics = checkpoint.metadata.metrics.unwrap();
        assert_eq!(loaded_metrics.mse, metrics.mse);
        assert_eq!(loaded_metrics.mae, metrics.mae);
        assert_eq!(loaded_metrics.rmse, metrics.rmse);

        // Cleanup
        std::fs::remove_file(&checkpoint_path).ok();
    }
}
