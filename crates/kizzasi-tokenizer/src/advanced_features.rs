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
///
/// # Training
///
/// [`HierarchicalTokenizer::new`] initializes every level's codebook with
/// Gaussian noise — a real codebook only exists after calling
/// [`HierarchicalTokenizer::fit`] (or [`HierarchicalTokenizer::set_codebooks`]
/// with externally-trained codebooks). [`HierarchicalTokenizer::encode_with_levels`]
/// and [`HierarchicalTokenizer::decode_hierarchical`] both return
/// [`TokenizerError::InvalidConfig`] until the tokenizer is trained,
/// mirroring [`crate::KMeansTokenizer`]'s "model not trained" check.
#[derive(Debug, Clone)]
pub struct HierarchicalTokenizer {
    config: HierarchicalConfig,
    /// Codebooks for each level (simplified - just centers)
    codebooks: Vec<Array2<f32>>,
    /// Whether `codebooks` holds trained centers (via `fit`/`set_codebooks`)
    /// rather than the random initialization `new` seeds them with.
    trained: bool,
}

impl HierarchicalTokenizer {
    /// Create a new hierarchical tokenizer.
    ///
    /// The returned tokenizer is **untrained**: codebooks are random
    /// Gaussian noise, a placeholder that only fixes each level's shape.
    /// Call [`HierarchicalTokenizer::fit`] before encoding/decoding.
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

        // Initialize random codebooks for each level (placeholder shape
        // only — see `fit`).
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

        Ok(Self {
            config,
            codebooks,
            trained: false,
        })
    }

    /// Train every level's codebook via residual k-means.
    ///
    /// Level 0's codebook is fit on `data` directly (k-means++ init followed
    /// by Lloyd iterations); if `config.use_residual` is set, each
    /// subsequent level is fit on the residual left after subtracting the
    /// nearest level-0..level-1 codeword from every point — the same
    /// residual-coding scheme [`HierarchicalTokenizer::encode_with_levels`]
    /// applies at inference time, so a signal encoded then decoded with the
    /// resulting codebooks reconstructs the training-set structure it was
    /// fit on.
    ///
    /// # Errors
    /// Returns an error if `data` is empty, if any point's length doesn't
    /// match the tokenizer's `embed_dim`, or if any level has fewer data
    /// points than its configured codebook size.
    pub fn fit(
        &mut self,
        data: &[Array1<f32>],
        max_iterations: usize,
        tolerance: f32,
    ) -> TokenizerResult<()> {
        if data.is_empty() {
            return Err(TokenizerError::InvalidConfig(
                "No training data".to_string(),
            ));
        }

        let embed_dim = self.codebooks[0].shape()[1];
        for (i, point) in data.iter().enumerate() {
            if point.len() != embed_dim {
                return Err(TokenizerError::dim_mismatch(
                    embed_dim,
                    point.len(),
                    format!("HierarchicalTokenizer::fit: data[{i}]"),
                ));
            }
        }

        let mut residuals: Vec<Array1<f32>> = data.to_vec();
        let mut trained_codebooks = Vec::with_capacity(self.config.num_levels);

        for level in 0..self.config.num_levels {
            let codebook_size = self.config.codebook_sizes[level];
            if residuals.len() < codebook_size {
                return Err(TokenizerError::InvalidConfig(format!(
                    "level {level}: only {} data points for {codebook_size} codes",
                    residuals.len()
                )));
            }

            let codebook = Self::kmeans(&residuals, codebook_size, max_iterations, tolerance);

            if self.config.use_residual && level + 1 < self.config.num_levels {
                for point in residuals.iter_mut() {
                    let best = Self::nearest_row(&codebook, point);
                    let code = codebook.row(best);
                    for (r, &c) in point.iter_mut().zip(code.iter()) {
                        *r -= c;
                    }
                }
            }

            trained_codebooks.push(codebook);
        }

        self.codebooks = trained_codebooks;
        self.trained = true;
        Ok(())
    }

    /// Run k-means (k-means++ initialization + Lloyd iterations) on `data`,
    /// returning a `[k, dim]` codebook.
    fn kmeans(
        data: &[Array1<f32>],
        k: usize,
        max_iterations: usize,
        tolerance: f32,
    ) -> Array2<f32> {
        let dim = data[0].len();
        let n = data.len();
        let mut rng = thread_rng();

        // k-means++ initialization.
        let mut centroids = Array2::<f32>::zeros((k, dim));
        centroids
            .row_mut(0)
            .assign(&data[rng.random_range(0..n)].view());

        let mut min_dist = vec![f32::INFINITY; n];
        for c in 1..k {
            let prev = centroids.row(c - 1).to_owned();
            for (i, point) in data.iter().enumerate() {
                let d: f32 = point
                    .iter()
                    .zip(prev.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                if d < min_dist[i] {
                    min_dist[i] = d;
                }
            }

            let total: f32 = min_dist.iter().sum();
            let chosen = if total <= 0.0 {
                c % n
            } else {
                let mut threshold = rng.random::<f32>() * total;
                let mut picked = n - 1;
                for (i, &d) in min_dist.iter().enumerate() {
                    threshold -= d;
                    if threshold <= 0.0 {
                        picked = i;
                        break;
                    }
                }
                picked
            };
            centroids.row_mut(c).assign(&data[chosen].view());
        }

        // Lloyd iterations.
        for _ in 0..max_iterations {
            let assignments: Vec<usize> = data
                .iter()
                .map(|p| Self::nearest_row(&centroids, p))
                .collect();

            let mut new_centroids = Array2::<f32>::zeros((k, dim));
            let mut counts = vec![0usize; k];
            for (point, &a) in data.iter().zip(assignments.iter()) {
                for (j, &v) in point.iter().enumerate() {
                    new_centroids[[a, j]] += v;
                }
                counts[a] += 1;
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..dim {
                        new_centroids[[c, j]] /= counts[c] as f32;
                    }
                } else {
                    // Keep empty clusters at their previous position rather
                    // than collapsing them to the origin.
                    for j in 0..dim {
                        new_centroids[[c, j]] = centroids[[c, j]];
                    }
                }
            }

            let change: f32 = new_centroids
                .iter()
                .zip(centroids.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            centroids = new_centroids;
            if change < tolerance {
                break;
            }
        }

        centroids
    }

    /// Index of `codebook`'s row nearest to `point` in squared Euclidean distance.
    fn nearest_row(codebook: &Array2<f32>, point: &Array1<f32>) -> usize {
        let mut best = 0;
        let mut best_dist = f32::INFINITY;
        for (idx, row) in codebook.outer_iter().enumerate() {
            let dist: f32 = point
                .iter()
                .zip(row.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best = idx;
            }
        }
        best
    }

    /// Whether the tokenizer has been trained via [`HierarchicalTokenizer::fit`]
    /// or [`HierarchicalTokenizer::set_codebooks`].
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Read-only access to the per-level codebooks.
    pub fn codebooks(&self) -> &[Array2<f32>] {
        &self.codebooks
    }

    /// Replace the per-level codebooks with externally-trained ones (e.g.
    /// loaded from disk), marking the tokenizer as trained.
    ///
    /// # Errors
    /// Returns an error unless `codebooks.len() == config.num_levels` and
    /// every level's shape is `[codebook_sizes[level], embed_dim]` (matching
    /// the embedding dimension the tokenizer was constructed with).
    pub fn set_codebooks(&mut self, codebooks: Vec<Array2<f32>>) -> TokenizerResult<()> {
        if codebooks.len() != self.config.num_levels {
            return Err(TokenizerError::dim_mismatch(
                self.config.num_levels,
                codebooks.len(),
                "HierarchicalTokenizer::set_codebooks: number of levels",
            ));
        }

        let embed_dim = self.codebooks[0].shape()[1];
        for (level, codebook) in codebooks.iter().enumerate() {
            let expected = (self.config.codebook_sizes[level], embed_dim);
            if codebook.shape() != [expected.0, expected.1] {
                return Err(TokenizerError::InvalidConfig(format!(
                    "level {level}: expected codebook shape {:?}, got {:?}",
                    expected,
                    codebook.shape()
                )));
            }
        }

        self.codebooks = codebooks;
        self.trained = true;
        Ok(())
    }

    /// Encode using specified number of levels (for variable bitrate)
    ///
    /// # Errors
    /// Returns [`TokenizerError::InvalidConfig`] if the tokenizer has not
    /// been trained yet (see [`HierarchicalTokenizer::fit`]): encoding
    /// against the random Gaussian codebooks `new` seeds would be a
    /// nearest-neighbour match against noise, not a meaningful code.
    pub fn encode_with_levels(
        &self,
        signal: &Array1<f32>,
        num_levels: usize,
    ) -> TokenizerResult<Vec<usize>> {
        if !self.trained {
            return Err(TokenizerError::InvalidConfig(
                "HierarchicalTokenizer not trained: call fit() or set_codebooks() first".into(),
            ));
        }

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
    ///
    /// # Errors
    /// Returns [`TokenizerError::InvalidConfig`] if the tokenizer has not
    /// been trained yet (see [`HierarchicalTokenizer::fit`]).
    pub fn decode_hierarchical(&self, indices: &[usize]) -> TokenizerResult<Array1<f32>> {
        if !self.trained {
            return Err(TokenizerError::InvalidConfig(
                "HierarchicalTokenizer not trained: call fit() or set_codebooks() first".into(),
            ));
        }

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

        // Get first level codebook entry. `ndarray::Array2::row` panics on
        // an out-of-range row, and `indices[0]` can come from a model's
        // argmax or from deserialized data — bounds-check it the same way
        // every subsequent level already is below.
        if indices[0] >= self.codebooks[0].shape()[0] {
            return Err(TokenizerError::decoding(
                "decoding",
                format!("Invalid index {} at level 0", indices[0]),
            ));
        }
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
        let mut tokenizer = HierarchicalTokenizer::new(8, config).unwrap();

        // Level 0 needs a codebook of size 256, so training needs at least
        // that many 8-dimensional points.
        let training_data: Vec<Array1<f32>> = (0..300)
            .map(|i| Array1::from_vec((0..8).map(|j| ((i * 8 + j) as f32 * 0.013).sin()).collect()))
            .collect();
        tokenizer.fit(&training_data, 10, 1e-3).unwrap();
        assert!(tokenizer.is_trained());

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

    /// Regression: `HierarchicalTokenizer::new` seeds every codebook with
    /// random Gaussian noise, not a trained model. `encode_with_levels`/
    /// `decode_hierarchical` must reject use before `fit`/`set_codebooks`,
    /// mirroring `KMeansTokenizer::encode`'s "model not trained" check,
    /// rather than silently nearest-neighbour-matching against noise.
    #[test]
    fn test_hierarchical_tokenizer_untrained_error() {
        let config = HierarchicalConfig::exponential(16, 2, 0.5);
        let tokenizer = HierarchicalTokenizer::new(4, config).unwrap();
        assert!(!tokenizer.is_trained());

        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(tokenizer.encode_with_levels(&signal, 1).is_err());
        assert!(tokenizer.decode_hierarchical(&[0]).is_err());
    }

    /// Regression: `fit` must actually move the codebooks away from their
    /// random Gaussian initialization towards the training data, and
    /// `set_codebooks` must validate shapes before accepting externally
    /// supplied codebooks.
    #[test]
    fn test_hierarchical_tokenizer_fit_changes_codebooks_and_set_codebooks_validates() {
        let config = HierarchicalConfig {
            num_levels: 1,
            codebook_sizes: vec![4],
            use_residual: false,
        };
        let mut tokenizer = HierarchicalTokenizer::new(2, config).unwrap();
        let codebook_before = tokenizer.codebooks()[0].clone();

        // Tightly clustered training data far from the ~N(0,1) random init.
        let training_data: Vec<Array1<f32>> = (0..40)
            .map(|i| Array1::from_vec(vec![100.0 + (i % 4) as f32, 100.0 + (i % 4) as f32]))
            .collect();
        tokenizer.fit(&training_data, 20, 1e-4).unwrap();
        assert!(tokenizer.is_trained());

        let codebook_after = &tokenizer.codebooks()[0];
        for (before, after) in codebook_before.iter().zip(codebook_after.iter()) {
            assert!(
                (after - before).abs() > 10.0,
                "fit should move codebook entries towards the training data (100+), \
                 before={before}, after={after}"
            );
            assert!(
                *after > 50.0,
                "codebook entry {after} should track training data near 100"
            );
        }

        // set_codebooks: wrong number of levels.
        assert!(tokenizer.set_codebooks(vec![]).is_err());
        // set_codebooks: wrong shape (codebook_size 3 != configured 4).
        assert!(tokenizer
            .set_codebooks(vec![Array2::zeros((3, 2))])
            .is_err());
        // set_codebooks: correct shape.
        assert!(tokenizer.set_codebooks(vec![Array2::zeros((4, 2))]).is_ok());
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
