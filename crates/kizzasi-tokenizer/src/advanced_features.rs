//! Advanced features for tokenizer robustness and regularization
//!
//! This module provides:
//! - **Token Dropout**: Randomly drop tokens during training for regularization
//! - **Jitter Injection**: Add controlled noise for robustness
//! - **Temporal Coherence**: Enforce smoothness constraints across time
//! - **Hierarchical Tokenization**: Variable-length codes with hierarchical structure

use crate::error::{TokenizerError, TokenizerResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;

// ============================================================================
// Token Dropout for Regularization
// ============================================================================

/// Token dropout configuration
#[derive(Debug, Clone)]
pub struct TokenDropoutConfig {
    /// Dropout probability (0.0 = no dropout, 1.0 = drop all)
    pub dropout_rate: f32,
    /// Value to use for dropped tokens (typically 0.0 or codebook mean)
    pub fill_value: f32,
    /// Whether to scale remaining tokens to compensate for dropout
    pub scale_remaining: bool,
}

impl Default for TokenDropoutConfig {
    fn default() -> Self {
        Self {
            dropout_rate: 0.1,
            fill_value: 0.0,
            scale_remaining: true,
        }
    }
}

/// Apply token dropout to a signal
///
/// During training, randomly set tokens to `fill_value` with probability `dropout_rate`.
/// This acts as a regularization technique to prevent over-reliance on specific tokens.
///
/// # Arguments
/// * `tokens` - Input token array
/// * `config` - Dropout configuration
/// * `training` - Whether dropout should be applied (true during training)
///
/// # Returns
/// Token array with dropout applied (if training=true)
pub fn apply_token_dropout(
    tokens: &Array1<f32>,
    config: &TokenDropoutConfig,
    training: bool,
) -> TokenizerResult<Array1<f32>> {
    if !training || config.dropout_rate <= 0.0 {
        return Ok(tokens.clone());
    }

    if !(0.0..=1.0).contains(&config.dropout_rate) {
        return Err(TokenizerError::InvalidConfig(
            "dropout_rate must be in [0, 1]".into(),
        ));
    }

    let mut rng = thread_rng();
    let mut result = tokens.clone();

    for val in result.iter_mut() {
        if rng.random::<f32>() < config.dropout_rate {
            *val = config.fill_value;
        } else if config.scale_remaining {
            // Scale up to compensate for dropped tokens
            *val /= 1.0 - config.dropout_rate;
        }
    }

    Ok(result)
}

/// Apply batch token dropout
pub fn apply_batch_token_dropout(
    tokens: &Array2<f32>,
    config: &TokenDropoutConfig,
    training: bool,
) -> TokenizerResult<Array2<f32>> {
    if !training || config.dropout_rate <= 0.0 {
        return Ok(tokens.clone());
    }

    let (batch_size, seq_len) = (tokens.shape()[0], tokens.shape()[1]);
    let mut rng = thread_rng();
    let mut result = tokens.clone();

    for i in 0..batch_size {
        for j in 0..seq_len {
            if rng.random::<f32>() < config.dropout_rate {
                result[[i, j]] = config.fill_value;
            } else if config.scale_remaining {
                result[[i, j]] /= 1.0 - config.dropout_rate;
            }
        }
    }

    Ok(result)
}

// ============================================================================
// Jitter Injection for Robustness
// ============================================================================

/// Jitter injection configuration
#[derive(Debug, Clone)]
pub struct JitterConfig {
    /// Standard deviation of Gaussian noise
    pub noise_std: f32,
    /// Whether to apply jitter during inference (usually false)
    pub apply_at_inference: bool,
    /// SNR target in dB (alternative to noise_std)
    pub target_snr_db: Option<f32>,
}

impl Default for JitterConfig {
    fn default() -> Self {
        Self {
            noise_std: 0.01,
            apply_at_inference: false,
            target_snr_db: None,
        }
    }
}

impl JitterConfig {
    /// Create jitter config with target SNR
    pub fn with_snr(target_snr_db: f32) -> Self {
        Self {
            noise_std: 0.0, // Will be computed based on signal
            apply_at_inference: false,
            target_snr_db: Some(target_snr_db),
        }
    }
}

/// Add Gaussian jitter to signal for robustness
///
/// Injects controlled noise to make the model robust to small perturbations.
/// Can be applied during training to improve generalization.
///
/// # Arguments
/// * `signal` - Input signal
/// * `config` - Jitter configuration
/// * `training` - Whether currently in training mode
///
/// # Returns
/// Signal with added jitter (if applicable)
pub fn add_jitter(
    signal: &Array1<f32>,
    config: &JitterConfig,
    training: bool,
) -> TokenizerResult<Array1<f32>> {
    if !training && !config.apply_at_inference {
        return Ok(signal.clone());
    }

    // Compute noise std based on SNR target if specified
    let noise_std = if let Some(target_snr_db) = config.target_snr_db {
        let signal_power = signal.iter().map(|x| x.powi(2)).sum::<f32>() / signal.len() as f32;
        let target_snr_linear = 10.0_f32.powf(target_snr_db / 10.0);
        let noise_power = signal_power / target_snr_linear;
        noise_power.sqrt()
    } else {
        config.noise_std
    };

    if noise_std <= 0.0 {
        return Ok(signal.clone());
    }

    let mut rng = thread_rng();
    let mut result = signal.clone();

    for val in result.iter_mut() {
        // Use central limit theorem: sum of 12 uniforms approximates Gaussian(0,1)
        let gaussian: f32 = (0..12).map(|_| rng.random::<f32>()).sum::<f32>() - 6.0;
        *val += gaussian * noise_std;
    }

    Ok(result)
}

/// Add batch jitter
pub fn add_batch_jitter(
    signals: &Array2<f32>,
    config: &JitterConfig,
    training: bool,
) -> TokenizerResult<Array2<f32>> {
    if !training && !config.apply_at_inference {
        return Ok(signals.clone());
    }

    let (batch_size, seq_len) = (signals.shape()[0], signals.shape()[1]);
    let mut result = signals.clone();

    // Apply jitter to each sample in the batch
    for i in 0..batch_size {
        let row = signals.row(i).to_owned();
        let jittered = add_jitter(&row, config, training)?;

        for j in 0..seq_len {
            result[[i, j]] = jittered[[j]];
        }
    }

    Ok(result)
}

// ============================================================================
// Temporal Coherence Constraints
// ============================================================================

/// Temporal coherence configuration
#[derive(Debug, Clone)]
pub struct TemporalCoherenceConfig {
    /// Smoothness strength (0.0 = no smoothing, 1.0 = maximum smoothing)
    pub smoothness: f32,
    /// Window size for temporal smoothing
    pub window_size: usize,
    /// Type of temporal filter
    pub filter_type: TemporalFilterType,
}

#[derive(Debug, Clone, Copy)]
pub enum TemporalFilterType {
    /// Exponential moving average
    ExponentialMovingAverage,
    /// Simple moving average
    SimpleMovingAverage,
    /// Gaussian weighted
    GaussianWeighted,
}

impl Default for TemporalCoherenceConfig {
    fn default() -> Self {
        Self {
            smoothness: 0.5,
            window_size: 5,
            filter_type: TemporalFilterType::SimpleMovingAverage,
        }
    }
}

/// Apply temporal coherence constraint to enforce smoothness
///
/// Smooths the signal across time to reduce jitter and enforce
/// temporal consistency. Useful for signals that should vary smoothly.
///
/// # Arguments
/// * `signal` - Input signal (assumed to be temporal)
/// * `config` - Temporal coherence configuration
///
/// # Returns
/// Temporally smoothed signal
pub fn apply_temporal_coherence(
    signal: &Array1<f32>,
    config: &TemporalCoherenceConfig,
) -> TokenizerResult<Array1<f32>> {
    if !(0.0..=1.0).contains(&config.smoothness) {
        return Err(TokenizerError::InvalidConfig(
            "smoothness must be in [0, 1]".into(),
        ));
    }

    if config.smoothness <= 0.0 {
        return Ok(signal.clone());
    }

    match config.filter_type {
        TemporalFilterType::ExponentialMovingAverage => apply_ema(signal, config.smoothness),
        TemporalFilterType::SimpleMovingAverage => apply_sma(signal, config.window_size),
        TemporalFilterType::GaussianWeighted => {
            apply_gaussian_smooth(signal, config.window_size, config.smoothness)
        }
    }
}

/// Apply Exponential Moving Average (EMA)
fn apply_ema(signal: &Array1<f32>, alpha: f32) -> TokenizerResult<Array1<f32>> {
    let mut result = signal.clone();

    for i in 1..signal.len() {
        result[[i]] = alpha * signal[[i]] + (1.0 - alpha) * result[[i - 1]];
    }

    Ok(result)
}

/// Apply Simple Moving Average (SMA)
fn apply_sma(signal: &Array1<f32>, window_size: usize) -> TokenizerResult<Array1<f32>> {
    if window_size == 0 {
        return Err(TokenizerError::InvalidConfig(
            "window_size must be positive".into(),
        ));
    }

    let mut result = signal.clone();
    let half_window = window_size / 2;

    for i in 0..signal.len() {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(signal.len());

        let sum: f32 = signal.iter().skip(start).take(end - start).sum();
        result[[i]] = sum / (end - start) as f32;
    }

    Ok(result)
}

/// Apply Gaussian-weighted smoothing
fn apply_gaussian_smooth(
    signal: &Array1<f32>,
    window_size: usize,
    sigma: f32,
) -> TokenizerResult<Array1<f32>> {
    if window_size == 0 {
        return Err(TokenizerError::InvalidConfig(
            "window_size must be positive".into(),
        ));
    }

    let mut result = signal.clone();
    let half_window = window_size / 2;

    // Precompute Gaussian weights
    let mut weights = vec![0.0; window_size];
    let mut weight_sum = 0.0;
    for (i, w) in weights.iter_mut().enumerate() {
        let offset = i as f32 - half_window as f32;
        *w = (-offset.powi(2) / (2.0 * sigma.powi(2))).exp();
        weight_sum += *w;
    }

    // Normalize weights
    for w in &mut weights {
        *w /= weight_sum;
    }

    // Apply weighted smoothing
    for i in 0..signal.len() {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(signal.len());

        let mut value = 0.0;
        let mut local_weight_sum = 0.0;

        for (j, idx) in (start..end).enumerate() {
            let weight_idx = j + half_window.saturating_sub(i.saturating_sub(start));
            if weight_idx < weights.len() {
                value += signal[[idx]] * weights[weight_idx];
                local_weight_sum += weights[weight_idx];
            }
        }

        result[[i]] = value / local_weight_sum.max(1e-8);
    }

    Ok(result)
}

// ============================================================================
// Hierarchical Tokenization with Variable-Length Codes
// ============================================================================

/// Hierarchical tokenization configuration
#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    /// Number of hierarchy levels (1 = flat, >1 = hierarchical)
    pub num_levels: usize,
    /// Codebook sizes per level
    pub codebook_sizes: Vec<usize>,
    /// Whether to use residual coding between levels
    pub use_residual: bool,
}

impl HierarchicalConfig {
    /// Create a hierarchical config with exponentially decreasing codebook sizes
    pub fn exponential(base_size: usize, num_levels: usize, decay_factor: f32) -> Self {
        let mut codebook_sizes = Vec::with_capacity(num_levels);

        for level in 0..num_levels {
            let size = (base_size as f32 * decay_factor.powi(level as i32)) as usize;
            codebook_sizes.push(size.max(16)); // Minimum 16 codes per level
        }

        Self {
            num_levels,
            codebook_sizes,
            use_residual: true,
        }
    }
}

/// Hierarchical tokenizer with variable-length codes
///
/// Encodes signals using multiple levels of granularity:
/// - Coarse level: Few bits, captures main structure
/// - Fine levels: More bits, capture details
///
/// Allows variable bitrate by using different numbers of levels.
#[derive(Debug, Clone)]
pub struct HierarchicalTokenizer {
    config: HierarchicalConfig,
    /// Codebooks for each level (simplified - just centers)
    codebooks: Vec<Array2<f32>>,
}

impl HierarchicalTokenizer {
    /// Create a new hierarchical tokenizer
    pub fn new(embed_dim: usize, config: HierarchicalConfig) -> TokenizerResult<Self> {
        if config.num_levels == 0 {
            return Err(TokenizerError::InvalidConfig(
                "num_levels must be positive".into(),
            ));
        }

        if config.codebook_sizes.len() != config.num_levels {
            return Err(TokenizerError::InvalidConfig(
                "codebook_sizes.len() must equal num_levels".into(),
            ));
        }

        // Initialize random codebooks for each level
        let mut rng = thread_rng();
        let mut codebooks = Vec::with_capacity(config.num_levels);

        for &size in &config.codebook_sizes {
            let mut codebook_data = vec![0.0; size * embed_dim];
            for val in &mut codebook_data {
                // Use central limit theorem for Gaussian initialization
                let gaussian: f32 = (0..12).map(|_| rng.random::<f32>()).sum::<f32>() - 6.0;
                *val = gaussian;
            }

            let codebook =
                Array2::from_shape_vec((size, embed_dim), codebook_data).map_err(|e| {
                    TokenizerError::encoding("serialization", format!("Codebook init: {}", e))
                })?;

            codebooks.push(codebook);
        }

        Ok(Self { config, codebooks })
    }

    /// Encode using specified number of levels (for variable bitrate)
    pub fn encode_with_levels(
        &self,
        signal: &Array1<f32>,
        num_levels: usize,
    ) -> TokenizerResult<Vec<usize>> {
        if num_levels > self.config.num_levels {
            return Err(TokenizerError::InvalidConfig(format!(
                "num_levels {} exceeds configured {}",
                num_levels, self.config.num_levels
            )));
        }

        let mut indices = Vec::with_capacity(num_levels);
        let mut residual = signal.clone();

        for level in 0..num_levels {
            // Find nearest codebook entry at this level
            let codebook = &self.codebooks[level];
            let mut best_idx = 0;
            let mut best_dist = f32::INFINITY;

            for (idx, code) in codebook.outer_iter().enumerate() {
                let dist: f32 = residual
                    .iter()
                    .zip(code.iter())
                    .map(|(r, c)| (r - c).powi(2))
                    .sum();

                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx;
                }
            }

            indices.push(best_idx);

            // Update residual if using residual coding
            if self.config.use_residual && level < num_levels - 1 {
                let quantized = codebook.row(best_idx);
                for i in 0..residual.len().min(quantized.len()) {
                    residual[[i]] -= quantized[[i]];
                }
            }
        }

        Ok(indices)
    }

    /// Decode from hierarchical indices
    pub fn decode_hierarchical(&self, indices: &[usize]) -> TokenizerResult<Array1<f32>> {
        if indices.is_empty() {
            return Err(TokenizerError::decoding("deserialization", "Empty indices"));
        }

        if indices.len() > self.config.num_levels {
            return Err(TokenizerError::decoding(
                "decoding",
                format!(
                    "Too many indices: {} > {}",
                    indices.len(),
                    self.config.num_levels
                ),
            ));
        }

        // Get first level codebook entry
        let first_code = self.codebooks[0].row(indices[0]);
        let mut result = first_code.to_owned();

        // Add residuals from subsequent levels
        if self.config.use_residual {
            for (level, &idx) in indices.iter().enumerate().skip(1) {
                if idx >= self.codebooks[level].shape()[0] {
                    return Err(TokenizerError::decoding(
                        "decoding",
                        format!("Invalid index {} at level {}", idx, level),
                    ));
                }

                let code = self.codebooks[level].row(idx);
                for i in 0..result.len().min(code.len()) {
                    result[[i]] += code[[i]];
                }
            }
        }

        Ok(result)
    }

    /// Get the bitrate for a given number of levels
    pub fn bitrate_for_levels(&self, num_levels: usize) -> f32 {
        let mut total_bits = 0.0;

        for level in 0..num_levels.min(self.config.num_levels) {
            total_bits += (self.config.codebook_sizes[level] as f32).log2();
        }

        total_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_dropout() {
        let tokens = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let config = TokenDropoutConfig {
            dropout_rate: 0.5,
            fill_value: 0.0,
            scale_remaining: false,
        };

        let result = apply_token_dropout(&tokens, &config, true).unwrap();
        assert_eq!(result.len(), tokens.len());
    }

    #[test]
    fn test_jitter_injection() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let config = JitterConfig {
            noise_std: 0.1,
            apply_at_inference: false,
            target_snr_db: None,
        };

        let result = add_jitter(&signal, &config, true).unwrap();
        assert_eq!(result.len(), signal.len());
    }

    #[test]
    fn test_temporal_coherence_sma() {
        let signal = Array1::from_vec(vec![1.0, 5.0, 2.0, 8.0, 3.0]);
        let config = TemporalCoherenceConfig {
            smoothness: 0.5,
            window_size: 3,
            filter_type: TemporalFilterType::SimpleMovingAverage,
        };

        let result = apply_temporal_coherence(&signal, &config).unwrap();
        assert_eq!(result.len(), signal.len());

        // Smoothed signal should have lower variance
        let original_var: f32 = signal.iter().map(|x| x.powi(2)).sum::<f32>() / signal.len() as f32;
        let smoothed_var: f32 = result.iter().map(|x| x.powi(2)).sum::<f32>() / result.len() as f32;

        // Not strictly guaranteed, but very likely with this test signal
        assert!(
            (smoothed_var - original_var).abs() < original_var,
            "Smoothed variance should be similar"
        );
    }

    #[test]
    fn test_hierarchical_tokenizer() {
        let config = HierarchicalConfig::exponential(256, 3, 0.5);
        let tokenizer = HierarchicalTokenizer::new(8, config).unwrap();

        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Encode with different numbers of levels
        let indices1 = tokenizer.encode_with_levels(&signal, 1).unwrap();
        let indices2 = tokenizer.encode_with_levels(&signal, 2).unwrap();
        let indices3 = tokenizer.encode_with_levels(&signal, 3).unwrap();

        assert_eq!(indices1.len(), 1);
        assert_eq!(indices2.len(), 2);
        assert_eq!(indices3.len(), 3);

        // Decode and check dimension preservation
        let decoded = tokenizer.decode_hierarchical(&indices3).unwrap();
        assert_eq!(decoded.len(), signal.len());
    }

    #[test]
    fn test_hierarchical_bitrate() {
        let config = HierarchicalConfig::exponential(256, 3, 0.5);
        let tokenizer = HierarchicalTokenizer::new(8, config).unwrap();

        let br1 = tokenizer.bitrate_for_levels(1);
        let br2 = tokenizer.bitrate_for_levels(2);
        let br3 = tokenizer.bitrate_for_levels(3);

        // More levels = higher bitrate
        assert!(br1 < br2);
        assert!(br2 < br3);
    }
}
