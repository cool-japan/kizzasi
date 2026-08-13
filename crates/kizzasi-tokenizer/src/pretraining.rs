//! Self-supervised pre-training for tokenizers.
//!
//! This module implements various self-supervised learning objectives for
//! pre-training signal tokenizers without requiring labeled data.
//!
//! # Supported Methods
//!
//! - **Masked Signal Modeling (MSM)**: Predict masked portions of the signal
//! - **Contrastive Learning**: Learn representations by contrasting similar/dissimilar segments
//! - **Temporal Prediction**: Predict future signal segments from past context
//! - **Denoising**: Reconstruct clean signals from noisy inputs
//!
//! # Example
//!
//! ```
//! use kizzasi_tokenizer::{MaskedSignalModeling, MSMConfig};
//! use scirs2_core::ndarray::Array1;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = MSMConfig {
//!     mask_ratio: 0.15,
//!     mask_length: 16,
//!     learning_rate: 0.001,
//!     ..Default::default()
//! };
//!
//! let mut msm = MaskedSignalModeling::new(config)?;
//! let signals = vec![Array1::linspace(0.0, 1.0, 256); 32];
//! msm.pretrain(&signals, 10)?;
//! # Ok(())
//! # }
//! ```

use crate::error::{TokenizerError, TokenizerResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rngs::StdRng, Random};
use serde::{Deserialize, Serialize};

/// Configuration for Masked Signal Modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MSMConfig {
    /// Ratio of signal to mask (0.0 to 1.0)
    pub mask_ratio: f32,
    /// Length of each masked segment
    pub mask_length: usize,
    /// Signal dimension
    pub signal_dim: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
}

impl Default for MSMConfig {
    fn default() -> Self {
        Self {
            mask_ratio: 0.15,
            mask_length: 16,
            signal_dim: 256,
            embed_dim: 128,
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
        }
    }
}

impl MSMConfig {
    /// Validate the configuration
    pub fn validate(&self) -> TokenizerResult<()> {
        if !(0.0..=1.0).contains(&self.mask_ratio) {
            return Err(TokenizerError::invalid_input(
                "mask_ratio must be in [0.0, 1.0]",
                "MSMConfig::validate",
            ));
        }
        if self.mask_length == 0 {
            return Err(TokenizerError::invalid_input(
                "mask_length must be positive",
                "MSMConfig::validate",
            ));
        }
        if self.signal_dim == 0 || self.embed_dim == 0 {
            return Err(TokenizerError::invalid_input(
                "signal_dim and embed_dim must be positive",
                "MSMConfig::validate",
            ));
        }
        if !(0.0..1.0).contains(&self.learning_rate) {
            return Err(TokenizerError::invalid_input(
                "learning_rate must be in (0.0, 1.0)",
                "MSMConfig::validate",
            ));
        }
        if self.epochs == 0 || self.batch_size == 0 {
            return Err(TokenizerError::invalid_input(
                "epochs and batch_size must be positive",
                "MSMConfig::validate",
            ));
        }
        Ok(())
    }
}

/// Masked Signal Modeling pre-trainer
///
/// Learns signal representations by predicting masked portions of the input.
#[derive(Debug)]
pub struct MaskedSignalModeling {
    /// Configuration
    config: MSMConfig,
    /// Encoder weights [signal_dim, embed_dim]
    encoder: Array2<f32>,
    /// Decoder weights [embed_dim, signal_dim]
    decoder: Array2<f32>,
    /// Random number generator
    rng: Random<StdRng>,
}

impl MaskedSignalModeling {
    /// Create a new MSM pre-trainer
    pub fn new(config: MSMConfig) -> TokenizerResult<Self> {
        config.validate()?;

        let mut rng = Random::seed(45);

        // Xavier initialization
        let encoder_scale = (2.0 / (config.signal_dim + config.embed_dim) as f32).sqrt();
        let decoder_scale = (2.0 / (config.embed_dim + config.signal_dim) as f32).sqrt();

        let encoder =
            Self::init_weights(config.signal_dim, config.embed_dim, encoder_scale, &mut rng);
        let decoder =
            Self::init_weights(config.embed_dim, config.signal_dim, decoder_scale, &mut rng);

        Ok(Self {
            config,
            encoder,
            decoder,
            rng,
        })
    }

    /// Initialize weights with Xavier uniform distribution
    fn init_weights(rows: usize, cols: usize, scale: f32, rng: &mut Random<StdRng>) -> Array2<f32> {
        let mut weights = Array2::zeros((rows, cols));
        for val in weights.iter_mut() {
            *val = (rng.gen_range(-1.0..1.0)) * scale;
        }
        weights
    }

    /// Create a mask for the signal
    ///
    /// Returns a boolean array where true indicates masked positions
    fn create_mask(&mut self, signal_len: usize) -> Array1<bool> {
        let mut mask = Array1::from_elem(signal_len, false);
        let num_masks = ((signal_len as f32 * self.config.mask_ratio)
            / self.config.mask_length as f32) as usize;

        for _ in 0..num_masks {
            let start = (self.rng.gen_range(0.0..1.0)
                * (signal_len - self.config.mask_length) as f32) as usize;
            let end = (start + self.config.mask_length).min(signal_len);
            for i in start..end {
                mask[i] = true;
            }
        }

        mask
    }

    /// Apply mask to signal (replace masked values with zeros)
    fn apply_mask(&self, signal: &Array1<f32>, mask: &Array1<bool>) -> Array1<f32> {
        signal
            .iter()
            .zip(mask.iter())
            .map(|(&val, &is_masked)| if is_masked { 0.0 } else { val })
            .collect()
    }

    /// Forward pass: encode and decode
    fn forward(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        // Encode: signal -> embedding
        let mut embedding = Array1::zeros(self.config.embed_dim);
        for j in 0..self.config.embed_dim {
            let mut sum = 0.0;
            for i in 0..self.config.signal_dim.min(signal.len()) {
                sum += signal[i] * self.encoder[[i, j]];
            }
            embedding[j] = sum;
        }

        // Apply ReLU activation
        embedding.mapv_inplace(|x| x.max(0.0));

        // Decode: embedding -> reconstructed signal
        let mut reconstructed = Array1::zeros(self.config.signal_dim);
        for i in 0..self.config.signal_dim {
            let mut sum = 0.0;
            for j in 0..self.config.embed_dim {
                sum += embedding[j] * self.decoder[[j, i]];
            }
            reconstructed[i] = sum;
        }

        Ok(reconstructed)
    }

    /// Compute MSE loss between target and prediction
    fn compute_loss(
        &self,
        target: &Array1<f32>,
        prediction: &Array1<f32>,
        mask: &Array1<bool>,
    ) -> f32 {
        let mut loss = 0.0;
        let mut count = 0;

        for i in 0..target.len().min(prediction.len()).min(mask.len()) {
            if mask[i] {
                let diff = target[i] - prediction[i];
                loss += diff * diff;
                count += 1;
            }
        }

        if count > 0 {
            loss / count as f32
        } else {
            0.0
        }
    }

    /// Pre-train on a dataset of signals
    pub fn pretrain(
        &mut self,
        signals: &[Array1<f32>],
        num_epochs: usize,
    ) -> TokenizerResult<Vec<f32>> {
        let mut losses = Vec::new();

        for epoch in 0..num_epochs {
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            // Process each signal
            for signal in signals {
                if signal.len() != self.config.signal_dim {
                    continue; // Skip signals with wrong dimension
                }

                // Create mask
                let mask = self.create_mask(signal.len());

                // Apply mask
                let masked_signal = self.apply_mask(signal, &mask);

                // Forward pass
                let reconstructed = self.forward(&masked_signal)?;

                // Compute loss (only on masked positions)
                let loss = self.compute_loss(signal, &reconstructed, &mask);
                epoch_loss += loss;
                num_batches += 1;

                // Backward pass (simplified gradient descent)
                self.update_weights(signal, &masked_signal, &reconstructed, &mask)?;
            }

            if num_batches > 0 {
                epoch_loss /= num_batches as f32;
                losses.push(epoch_loss);

                if epoch % 10 == 0 {
                    tracing::debug!("Epoch {}: Loss = {:.6}", epoch, epoch_loss);
                }
            }
        }

        Ok(losses)
    }

    /// Update weights using gradient descent (simplified)
    fn update_weights(
        &mut self,
        target: &Array1<f32>,
        input: &Array1<f32>,
        output: &Array1<f32>,
        mask: &Array1<bool>,
    ) -> TokenizerResult<()> {
        let lr = self.config.learning_rate;

        // Compute output error (only for masked positions)
        let mut output_error = Array1::zeros(self.config.signal_dim);
        for i in 0..self.config.signal_dim.min(output.len()).min(target.len()) {
            if i < mask.len() && mask[i] {
                output_error[i] = output[i] - target[i];
            }
        }

        // Compute embedding
        let mut embedding = Array1::zeros(self.config.embed_dim);
        for j in 0..self.config.embed_dim {
            let mut sum = 0.0;
            for i in 0..self.config.signal_dim.min(input.len()) {
                sum += input[i] * self.encoder[[i, j]];
            }
            embedding[j] = sum.max(0.0); // ReLU
        }

        // Snapshot the decoder *before* updating it. Backpropagation needs
        // the hidden error computed from the pre-update weights W, not from
        // W - lr*grad(W) — reading `self.decoder` after the update below
        // would follow a perturbed direction whose deviation from the true
        // gradient grows with `learning_rate` (user-controlled, unbounded).
        let decoder_before_update = self.decoder.clone();

        // Update decoder weights
        for j in 0..self.config.embed_dim {
            for i in 0..self.config.signal_dim {
                let gradient = output_error[i] * embedding[j];
                self.decoder[[j, i]] -= lr * gradient;
            }
        }

        // Compute hidden error using the pre-update decoder weights.
        let mut hidden_error = Array1::zeros(self.config.embed_dim);
        for j in 0..self.config.embed_dim {
            let mut sum = 0.0;
            for i in 0..self.config.signal_dim {
                sum += output_error[i] * decoder_before_update[[j, i]];
            }
            // ReLU derivative (0 if embedding[j] <= 0, else 1)
            hidden_error[j] = if embedding[j] > 0.0 { sum } else { 0.0 };
        }

        // Update encoder weights
        for i in 0..self.config.signal_dim.min(input.len()) {
            for j in 0..self.config.embed_dim {
                let gradient = hidden_error[j] * input[i];
                self.encoder[[i, j]] -= lr * gradient;
            }
        }

        Ok(())
    }

    /// Get the learned encoder weights
    pub fn encoder_weights(&self) -> &Array2<f32> {
        &self.encoder
    }

    /// Get the learned decoder weights
    pub fn decoder_weights(&self) -> &Array2<f32> {
        &self.decoder
    }
}

/// Configuration for Contrastive Learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastiveConfig {
    /// Embedding dimension
    pub embed_dim: usize,
    /// Temperature for contrastive loss
    pub temperature: f32,
    /// Augmentation noise std
    pub aug_noise_std: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Number of negative samples
    pub num_negatives: usize,
}

impl Default for ContrastiveConfig {
    fn default() -> Self {
        Self {
            embed_dim: 128,
            temperature: 0.07,
            aug_noise_std: 0.1,
            learning_rate: 0.001,
            num_negatives: 16,
        }
    }
}

/// Contrastive learning pre-trainer
///
/// Learns representations by maximizing agreement between augmented views
/// of the same signal while minimizing agreement with different signals.
#[derive(Debug)]
pub struct ContrastiveLearning {
    /// Configuration
    config: ContrastiveConfig,
    /// Encoder weights
    encoder: Array2<f32>,
    /// Random number generator
    rng: Random<StdRng>,
}

impl ContrastiveLearning {
    /// Create a new contrastive learning pre-trainer
    pub fn new(signal_dim: usize, config: ContrastiveConfig) -> Self {
        let mut rng = Random::seed(46);
        let scale = (2.0 / (signal_dim + config.embed_dim) as f32).sqrt();

        let mut encoder = Array2::zeros((signal_dim, config.embed_dim));
        for val in encoder.iter_mut() {
            *val = (rng.gen_range(-1.0..1.0)) * scale;
        }

        Self {
            config,
            encoder,
            rng,
        }
    }

    /// Apply data augmentation (add noise)
    fn augment(&mut self, signal: &Array1<f32>) -> Array1<f32> {
        signal.mapv(|x| {
            let noise = (self.rng.gen_range(-1.0..1.0)) * self.config.aug_noise_std;
            x + noise
        })
    }

    /// Encode signal to embedding
    fn encode(&self, signal: &Array1<f32>) -> Array1<f32> {
        let mut embedding = Array1::zeros(self.config.embed_dim);
        for j in 0..self.config.embed_dim {
            let mut sum = 0.0;
            for i in 0..signal.len().min(self.encoder.nrows()) {
                sum += signal[i] * self.encoder[[i, j]];
            }
            embedding[j] = sum;
        }

        // L2 normalization
        let norm = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            embedding /= norm;
        }
        embedding
    }

    /// Cosine similarity between two embeddings.
    ///
    /// Computed as a plain dot product, which equals the cosine only because
    /// every argument is an [`Self::encode`] output and `encode` L2-normalizes
    /// before returning. Passing an un-normalized vector here yields a dot
    /// product, not a cosine — this is a private helper precisely so that
    /// precondition stays enforceable by inspection of its two call sites in
    /// [`Self::contrastive_loss`].
    ///
    /// A zero embedding (which `encode` returns when the pre-normalization
    /// norm is 0) similarity-scores as 0 rather than NaN, so a degenerate
    /// signal cannot poison the NT-Xent softmax.
    fn cosine_similarity(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        debug_assert_eq!(
            a.len(),
            b.len(),
            "cosine_similarity operands must share embed_dim; `zip` would \
             otherwise silently truncate to the shorter one"
        );
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// Compute contrastive loss (NT-Xent)
    pub fn contrastive_loss(&mut self, signals: &[Array1<f32>]) -> TokenizerResult<f32> {
        if signals.len() < 2 {
            return Ok(0.0);
        }

        let mut total_loss = 0.0;
        let mut count = 0;

        for i in 0..signals.len() {
            // Create two augmented views of the same signal
            let view1 = self.augment(&signals[i]);
            let view2 = self.augment(&signals[i]);

            let z1 = self.encode(&view1);
            let z2 = self.encode(&view2);

            // Positive pair similarity
            let pos_sim = self.cosine_similarity(&z1, &z2) / self.config.temperature;

            // Negative pairs (from other signals)
            let mut neg_sims = Vec::new();
            for (j, signal) in signals.iter().enumerate() {
                if i != j {
                    let neg_view = self.augment(signal);
                    let z_neg = self.encode(&neg_view);
                    let neg_sim = self.cosine_similarity(&z1, &z_neg) / self.config.temperature;
                    neg_sims.push(neg_sim);

                    if neg_sims.len() >= self.config.num_negatives {
                        break;
                    }
                }
            }

            // NT-Xent loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            let pos_exp = pos_sim.exp();
            let neg_sum: f32 = neg_sims.iter().map(|&x| x.exp()).sum();
            let loss = -(pos_exp / (pos_exp + neg_sum)).ln();

            total_loss += loss;
            count += 1;
        }

        Ok(if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        })
    }

    /// Get encoder weights
    pub fn encoder_weights(&self) -> &Array2<f32> {
        &self.encoder
    }
}

/// Configuration for Temporal Prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPredictionConfig {
    /// Context window size
    pub context_size: usize,
    /// Prediction horizon
    pub prediction_size: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Learning rate
    pub learning_rate: f32,
}

impl Default for TemporalPredictionConfig {
    fn default() -> Self {
        Self {
            context_size: 64,
            prediction_size: 16,
            embed_dim: 128,
            learning_rate: 0.001,
        }
    }
}

/// Temporal prediction pre-trainer
///
/// Learns representations by predicting future signal segments from past context.
#[derive(Debug, Clone)]
pub struct TemporalPrediction {
    /// Configuration
    config: TemporalPredictionConfig,
    /// Context encoder weights
    context_encoder: Array2<f32>,
    /// Prediction head weights
    prediction_head: Array2<f32>,
}

impl TemporalPrediction {
    /// Create a new temporal prediction pre-trainer
    pub fn new(config: TemporalPredictionConfig) -> Self {
        let mut rng = Random::seed(47);

        let encoder_scale = (2.0 / (config.context_size + config.embed_dim) as f32).sqrt();
        let head_scale = (2.0 / (config.embed_dim + config.prediction_size) as f32).sqrt();

        let mut context_encoder = Array2::zeros((config.context_size, config.embed_dim));
        let mut prediction_head = Array2::zeros((config.embed_dim, config.prediction_size));

        for val in context_encoder.iter_mut() {
            *val = (rng.gen_range(-1.0..1.0)) * encoder_scale;
        }
        for val in prediction_head.iter_mut() {
            *val = (rng.gen_range(-1.0..1.0)) * head_scale;
        }

        Self {
            config,
            context_encoder,
            prediction_head,
        }
    }

    /// Predict future segment from context
    pub fn predict(&self, context: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if context.len() != self.config.context_size {
            return Err(TokenizerError::encoding(
                format!(
                    "Context size mismatch: expected {}, got {}",
                    self.config.context_size,
                    context.len()
                ),
                "TemporalPrediction::predict",
            ));
        }

        // Encode context
        let mut embedding = Array1::zeros(self.config.embed_dim);
        for j in 0..self.config.embed_dim {
            let mut sum = 0.0;
            for i in 0..self.config.context_size {
                sum += context[i] * self.context_encoder[[i, j]];
            }
            embedding[j] = sum.max(0.0); // ReLU
        }

        // Predict future
        let mut prediction = Array1::zeros(self.config.prediction_size);
        for i in 0..self.config.prediction_size {
            let mut sum = 0.0;
            for j in 0..self.config.embed_dim {
                sum += embedding[j] * self.prediction_head[[j, i]];
            }
            prediction[i] = sum;
        }

        Ok(prediction)
    }

    /// Get context encoder weights
    pub fn context_encoder_weights(&self) -> &Array2<f32> {
        &self.context_encoder
    }

    /// Get prediction head weights
    pub fn prediction_head_weights(&self) -> &Array2<f32> {
        &self.prediction_head
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_msm_config_validation() {
        let config = MSMConfig::default();
        assert!(config.validate().is_ok());

        let mut bad_config = config.clone();
        bad_config.mask_ratio = 1.5;
        assert!(bad_config.validate().is_err());

        let mut bad_config = config.clone();
        bad_config.learning_rate = 1.5;
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_msm_creation() {
        let config = MSMConfig::default();
        let msm = MaskedSignalModeling::new(config);
        assert!(msm.is_ok());
    }

    #[test]
    fn test_msm_create_mask() {
        let config = MSMConfig {
            mask_ratio: 0.2,
            mask_length: 10,
            ..Default::default()
        };
        let mut msm = MaskedSignalModeling::new(config).unwrap();

        let mask = msm.create_mask(100);
        assert_eq!(mask.len(), 100);

        // Count masked positions
        let num_masked = mask.iter().filter(|&&x| x).count();
        assert!(num_masked > 0 && num_masked < 100);
    }

    #[test]
    fn test_msm_apply_mask() {
        let config = MSMConfig::default();
        let msm = MaskedSignalModeling::new(config).unwrap();

        let signal = Array1::linspace(0.0, 1.0, 100);
        let mask = Array1::from_vec(vec![false; 50].into_iter().chain(vec![true; 50]).collect());

        let masked = msm.apply_mask(&signal, &mask);
        assert_eq!(masked.len(), 100);

        // First half should be unchanged, second half should be zero
        for i in 0..50 {
            assert!((masked[i] - signal[i]).abs() < 1e-6);
        }
        for i in 50..100 {
            assert_eq!(masked[i], 0.0);
        }
    }

    #[test]
    fn test_msm_forward() {
        let config = MSMConfig {
            signal_dim: 64,
            embed_dim: 32,
            ..Default::default()
        };
        let msm = MaskedSignalModeling::new(config).unwrap();

        let signal = Array1::linspace(0.0, 1.0, 64);
        let reconstructed = msm.forward(&signal);
        assert!(reconstructed.is_ok());

        let reconstructed = reconstructed.unwrap();
        assert_eq!(reconstructed.len(), 64);
    }

    #[test]
    fn test_msm_pretrain() {
        let config = MSMConfig {
            signal_dim: 32,
            embed_dim: 16,
            epochs: 5,
            ..Default::default()
        };
        let mut msm = MaskedSignalModeling::new(config).unwrap();

        let signals: Vec<Array1<f32>> = (0..10)
            .map(|i| Array1::linspace(i as f32, (i + 1) as f32, 32))
            .collect();

        let losses = msm.pretrain(&signals, 5);
        assert!(losses.is_ok());

        let losses = losses.unwrap();
        assert_eq!(losses.len(), 5);

        // Loss should generally decrease
        assert!(losses[4] <= losses[0] * 1.5); // Allow some variance
    }

    /// Regression: `update_weights` used to compute the hidden-layer error
    /// from the *already-updated* decoder weights (W - lr*grad) instead of
    /// the pre-update weights W that backpropagation requires, biasing the
    /// encoder update by an amount that grows with `learning_rate`. Uses a
    /// larger learning rate than `test_msm_pretrain` (which passes even
    /// with the bug, since its default `0.001` rate keeps the deviation
    /// tiny) so a wrong-direction encoder update would show up as loss
    /// failing to decrease.
    #[test]
    fn test_msm_pretrain_loss_decreases_with_larger_learning_rate() {
        let config = MSMConfig {
            signal_dim: 16,
            embed_dim: 8,
            mask_ratio: 0.5,
            mask_length: 4,
            learning_rate: 0.1,
            epochs: 20,
            ..Default::default()
        };
        let mut msm = MaskedSignalModeling::new(config).unwrap();

        // A simple, easily-reconstructable synthetic dataset: repeated
        // linear ramps.
        let signals: Vec<Array1<f32>> = (0..8).map(|_| Array1::linspace(0.0, 1.0, 16)).collect();

        let losses = msm.pretrain(&signals, 20).unwrap();
        assert_eq!(losses.len(), 20);
        assert!(losses.iter().all(|l| l.is_finite()));

        // The loss trend (comparing the mean of the first few epochs to the
        // mean of the last few) must be downward, not just "not much worse".
        let early: f32 = losses[..5].iter().sum::<f32>() / 5.0;
        let late: f32 = losses[15..].iter().sum::<f32>() / 5.0;
        assert!(
            late < early,
            "loss should trend downward over training: early={early}, late={late}, losses={losses:?}"
        );
    }

    #[test]
    fn test_contrastive_learning_creation() {
        let config = ContrastiveConfig::default();
        let cl = ContrastiveLearning::new(128, config);
        assert_eq!(cl.encoder.nrows(), 128);
    }

    #[test]
    fn test_contrastive_augment() {
        let config = ContrastiveConfig {
            aug_noise_std: 0.1,
            ..Default::default()
        };
        let mut cl = ContrastiveLearning::new(64, config);

        let signal = Array1::zeros(64);
        let augmented = cl.augment(&signal);
        assert_eq!(augmented.len(), 64);

        // Should have some noise
        let has_noise = augmented.iter().any(|&x| x != 0.0);
        assert!(has_noise);
    }

    #[test]
    fn test_contrastive_encode() {
        let config = ContrastiveConfig::default();
        let cl = ContrastiveLearning::new(64, config);

        let signal = Array1::linspace(0.0, 1.0, 64);
        let embedding = cl.encode(&signal);
        assert_eq!(embedding.len(), cl.config.embed_dim);

        // Check L2 normalization
        let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_contrastive_loss() {
        let config = ContrastiveConfig {
            num_negatives: 2,
            ..Default::default()
        };
        let mut cl = ContrastiveLearning::new(32, config);

        let signals: Vec<Array1<f32>> = (0..5)
            .map(|i| Array1::linspace(i as f32, (i + 1) as f32, 32))
            .collect();

        let loss = cl.contrastive_loss(&signals);
        assert!(loss.is_ok());

        let loss = loss.unwrap();
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn test_temporal_prediction_creation() {
        let config = TemporalPredictionConfig::default();
        let tp = TemporalPrediction::new(config);
        assert_eq!(tp.context_encoder.nrows(), tp.config.context_size);
    }

    #[test]
    fn test_temporal_prediction_predict() {
        let config = TemporalPredictionConfig {
            context_size: 32,
            prediction_size: 8,
            embed_dim: 16,
            ..Default::default()
        };
        let tp = TemporalPrediction::new(config);

        let context = Array1::linspace(0.0, 1.0, 32);
        let prediction = tp.predict(&context);
        assert!(prediction.is_ok());

        let prediction = prediction.unwrap();
        assert_eq!(prediction.len(), 8);
    }

    #[test]
    fn test_temporal_prediction_wrong_context_size() {
        let config = TemporalPredictionConfig {
            context_size: 32,
            ..Default::default()
        };
        let tp = TemporalPrediction::new(config);

        let wrong_context = Array1::linspace(0.0, 1.0, 16); // Wrong size
        let prediction = tp.predict(&wrong_context);
        assert!(prediction.is_err());
    }
}
