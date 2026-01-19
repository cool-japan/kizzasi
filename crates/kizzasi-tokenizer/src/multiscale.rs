//! Multi-scale (hierarchical) tokenization
//!
//! Processes signals at multiple temporal resolutions for better
//! representation of both local details and global structure.
//!
//! This is similar to hierarchical audio codecs (SoundStream, Encodec)
//! where lower levels capture fine details and higher levels capture
//! coarse structure.

use crate::error::{TokenizerError, TokenizerResult};
use crate::SignalTokenizer;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;

/// Configuration for a single scale level
#[derive(Debug, Clone)]
pub struct ScaleLevel {
    /// Downsampling factor relative to original signal
    downsample_factor: usize,
    /// Embedding dimension for this level
    embed_dim: usize,
    /// Input dimension (derived from downsample)
    input_dim: usize,
}

impl ScaleLevel {
    /// Create a new scale level
    pub fn new(downsample_factor: usize, embed_dim: usize, input_dim: usize) -> Self {
        Self {
            downsample_factor,
            embed_dim,
            input_dim,
        }
    }
}

/// Multi-scale hierarchical tokenizer
///
/// Processes signals at multiple temporal resolutions:
/// - Level 0: Full resolution (finest details)
/// - Level 1: 2x downsampled
/// - Level 2: 4x downsampled
/// - ...
///
/// Each level captures patterns at its corresponding scale.
#[derive(Debug, Clone)]
pub struct MultiScaleTokenizer {
    /// Encoder projections for each level
    encoders: Vec<Array2<f32>>,
    /// Decoder projections for each level
    decoders: Vec<Array2<f32>>,
    /// Scale configuration
    levels: Vec<ScaleLevel>,
    /// Original input dimension
    input_dim: usize,
    /// Pooling method for downsampling
    pool_method: PoolMethod,
    /// Upsampling method for reconstruction
    upsample_method: UpsampleMethod,
}

/// Method for downsampling signals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PoolMethod {
    /// Take every Nth sample
    Stride,
    /// Average pooling
    #[default]
    Average,
    /// Max pooling
    Max,
}

/// Method for upsampling signals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpsampleMethod {
    /// Repeat values (nearest neighbor)
    Repeat,
    /// Linear interpolation
    #[default]
    Linear,
}

impl MultiScaleTokenizer {
    /// Create a new multi-scale tokenizer with default scales
    ///
    /// Default: 3 levels with downsample factors 1, 2, 4
    pub fn new(input_dim: usize, embed_dim_per_level: usize) -> Self {
        Self::with_factors(input_dim, embed_dim_per_level, &[1, 2, 4])
    }

    /// Create with custom downsample factors
    pub fn with_factors(input_dim: usize, embed_dim_per_level: usize, factors: &[usize]) -> Self {
        let mut rng = thread_rng();
        let mut encoders = Vec::with_capacity(factors.len());
        let mut decoders = Vec::with_capacity(factors.len());
        let mut levels = Vec::with_capacity(factors.len());

        for &factor in factors {
            let level_input_dim = input_dim / factor;
            if level_input_dim == 0 {
                continue;
            }

            // Xavier initialization
            let enc_scale = (2.0 / (level_input_dim + embed_dim_per_level) as f32).sqrt();
            let encoder = Array2::from_shape_fn((level_input_dim, embed_dim_per_level), |_| {
                (rng.random::<f32>() - 0.5) * 2.0 * enc_scale
            });

            let dec_scale = (2.0 / (embed_dim_per_level + level_input_dim) as f32).sqrt();
            let decoder = Array2::from_shape_fn((embed_dim_per_level, level_input_dim), |_| {
                (rng.random::<f32>() - 0.5) * 2.0 * dec_scale
            });

            encoders.push(encoder);
            decoders.push(decoder);
            levels.push(ScaleLevel::new(
                factor,
                embed_dim_per_level,
                level_input_dim,
            ));
        }

        Self {
            encoders,
            decoders,
            levels,
            input_dim,
            pool_method: PoolMethod::default(),
            upsample_method: UpsampleMethod::default(),
        }
    }

    /// Set pooling method
    pub fn with_pool_method(mut self, method: PoolMethod) -> Self {
        self.pool_method = method;
        self
    }

    /// Set upsampling method
    pub fn with_upsample_method(mut self, method: UpsampleMethod) -> Self {
        self.upsample_method = method;
        self
    }

    /// Get number of levels
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get total embedding dimension across all levels
    pub fn total_embed_dim(&self) -> usize {
        self.levels.iter().map(|l| l.embed_dim).sum()
    }

    /// Downsample signal by given factor
    fn downsample(&self, signal: &Array1<f32>, factor: usize) -> Array1<f32> {
        if factor <= 1 {
            return signal.clone();
        }

        let new_len = signal.len() / factor;
        if new_len == 0 {
            return Array1::zeros(1);
        }

        match self.pool_method {
            PoolMethod::Stride => {
                Array1::from_vec((0..new_len).map(|i| signal[i * factor]).collect())
            }
            PoolMethod::Average => Array1::from_vec(
                (0..new_len)
                    .map(|i| {
                        let start = i * factor;
                        let end = (start + factor).min(signal.len());
                        signal.iter().skip(start).take(end - start).sum::<f32>()
                            / (end - start) as f32
                    })
                    .collect(),
            ),
            PoolMethod::Max => Array1::from_vec(
                (0..new_len)
                    .map(|i| {
                        let start = i * factor;
                        let end = (start + factor).min(signal.len());
                        signal
                            .iter()
                            .skip(start)
                            .take(end - start)
                            .cloned()
                            .fold(f32::NEG_INFINITY, f32::max)
                    })
                    .collect(),
            ),
        }
    }

    /// Upsample signal by given factor
    fn upsample(&self, signal: &Array1<f32>, factor: usize, target_len: usize) -> Array1<f32> {
        if factor <= 1 {
            return signal.clone();
        }

        match self.upsample_method {
            UpsampleMethod::Repeat => {
                let mut result = Vec::with_capacity(target_len);
                for &val in signal.iter() {
                    for _ in 0..factor {
                        if result.len() < target_len {
                            result.push(val);
                        }
                    }
                }
                // Pad if needed
                while result.len() < target_len {
                    result.push(*signal.last().unwrap_or(&0.0));
                }
                Array1::from_vec(result)
            }
            UpsampleMethod::Linear => {
                if signal.len() < 2 {
                    return Array1::from_elem(target_len, signal.get(0).copied().unwrap_or(0.0));
                }

                let mut result = Vec::with_capacity(target_len);
                for i in 0..target_len {
                    // Map target position to source position
                    let src_pos = i as f32 / factor as f32;
                    let src_idx = src_pos.floor() as usize;
                    let t = src_pos - src_idx as f32;

                    let val = if src_idx + 1 < signal.len() {
                        signal[src_idx] * (1.0 - t) + signal[src_idx + 1] * t
                    } else {
                        signal[signal.len() - 1]
                    };
                    result.push(val);
                }
                Array1::from_vec(result)
            }
        }
    }

    /// Encode at a specific level
    pub fn encode_level(&self, signal: &Array1<f32>, level: usize) -> TokenizerResult<Array1<f32>> {
        if level >= self.levels.len() {
            return Err(TokenizerError::InvalidConfig(format!(
                "Level {} out of range (0..{})",
                level,
                self.levels.len()
            )));
        }

        let factor = self.levels[level].downsample_factor;
        let downsampled = self.downsample(signal, factor);

        if downsampled.len() != self.levels[level].input_dim {
            // Resize to match expected dimension
            let mut resized = Array1::zeros(self.levels[level].input_dim);
            for i in 0..resized.len().min(downsampled.len()) {
                resized[i] = downsampled[i];
            }
            return Ok(resized.dot(&self.encoders[level]));
        }

        Ok(downsampled.dot(&self.encoders[level]))
    }

    /// Decode at a specific level
    pub fn decode_level(
        &self,
        embedding: &Array1<f32>,
        level: usize,
    ) -> TokenizerResult<Array1<f32>> {
        if level >= self.levels.len() {
            return Err(TokenizerError::InvalidConfig(format!(
                "Level {} out of range (0..{})",
                level,
                self.levels.len()
            )));
        }

        if embedding.len() != self.levels[level].embed_dim {
            return Err(TokenizerError::dim_mismatch(
                self.levels[level].embed_dim,
                embedding.len(),
                "dimension validation",
            ));
        }

        let decoded = embedding.dot(&self.decoders[level]);
        let factor = self.levels[level].downsample_factor;

        Ok(self.upsample(&decoded, factor, self.input_dim))
    }

    /// Encode all levels and concatenate embeddings
    pub fn encode_all(&self, signal: &Array1<f32>) -> TokenizerResult<Vec<Array1<f32>>> {
        let mut embeddings = Vec::with_capacity(self.levels.len());
        for level in 0..self.levels.len() {
            embeddings.push(self.encode_level(signal, level)?);
        }
        Ok(embeddings)
    }

    /// Decode all levels and combine
    pub fn decode_all(&self, embeddings: &[Array1<f32>]) -> TokenizerResult<Array1<f32>> {
        if embeddings.len() != self.levels.len() {
            return Err(TokenizerError::InvalidConfig(format!(
                "Expected {} embeddings, got {}",
                self.levels.len(),
                embeddings.len()
            )));
        }

        let mut result = Array1::zeros(self.input_dim);
        let weight = 1.0 / self.levels.len() as f32;

        for (level, embedding) in embeddings.iter().enumerate() {
            let decoded = self.decode_level(embedding, level)?;
            result = &result + &(&decoded * weight);
        }

        Ok(result)
    }

    /// Get concatenated encoding (all levels flattened)
    pub fn encode_concat(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let embeddings = self.encode_all(signal)?;
        let total_len: usize = embeddings.iter().map(|e| e.len()).sum();
        let mut result = Vec::with_capacity(total_len);
        for emb in embeddings {
            result.extend(emb.iter());
        }
        Ok(Array1::from_vec(result))
    }
}

impl SignalTokenizer for MultiScaleTokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if signal.len() != self.input_dim {
            return Err(TokenizerError::dim_mismatch(
                self.input_dim,
                signal.len(),
                "dimension validation",
            ));
        }
        self.encode_concat(signal)
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if tokens.len() != self.total_embed_dim() {
            return Err(TokenizerError::dim_mismatch(
                self.total_embed_dim(),
                tokens.len(),
                "dimension validation",
            ));
        }

        // Split tokens back into level embeddings
        let mut embeddings = Vec::with_capacity(self.levels.len());
        let mut offset = 0;
        for level in &self.levels {
            let end = offset + level.embed_dim;
            let embedding: Array1<f32> = Array1::from_vec(
                tokens
                    .iter()
                    .skip(offset)
                    .take(level.embed_dim)
                    .cloned()
                    .collect(),
            );
            embeddings.push(embedding);
            offset = end;
        }

        self.decode_all(&embeddings)
    }

    fn embed_dim(&self) -> usize {
        self.total_embed_dim()
    }

    fn vocab_size(&self) -> usize {
        0 // Continuous
    }
}

/// Pyramid tokenizer with residual connections
///
/// Each level encodes the residual from the previous level's reconstruction,
/// similar to residual VQ (RVQ) used in audio codecs.
#[derive(Debug, Clone)]
pub struct PyramidTokenizer {
    /// Base multi-scale tokenizer
    inner: MultiScaleTokenizer,
    /// Whether to use residual encoding
    use_residual: bool,
}

impl PyramidTokenizer {
    /// Create a new pyramid tokenizer
    pub fn new(input_dim: usize, embed_dim_per_level: usize, num_levels: usize) -> Self {
        // Generate factors: 1, 2, 4, 8, ...
        let factors: Vec<usize> = (0..num_levels).map(|i| 1 << i).collect();
        let inner = MultiScaleTokenizer::with_factors(input_dim, embed_dim_per_level, &factors);

        Self {
            inner,
            use_residual: true,
        }
    }

    /// Disable residual encoding (use independent levels)
    pub fn without_residual(mut self) -> Self {
        self.use_residual = false;
        self
    }

    /// Encode with residual pyramid
    pub fn encode_pyramid(&self, signal: &Array1<f32>) -> TokenizerResult<Vec<Array1<f32>>> {
        if !self.use_residual {
            return self.inner.encode_all(signal);
        }

        let mut embeddings = Vec::with_capacity(self.inner.num_levels());
        let mut residual = signal.clone();

        for level in 0..self.inner.num_levels() {
            let embedding = self.inner.encode_level(&residual, level)?;
            embeddings.push(embedding.clone());

            // Compute reconstruction and subtract from residual
            let reconstruction = self.inner.decode_level(&embedding, level)?;
            residual = &residual - &reconstruction;
        }

        Ok(embeddings)
    }

    /// Decode from pyramid embeddings
    pub fn decode_pyramid(&self, embeddings: &[Array1<f32>]) -> TokenizerResult<Array1<f32>> {
        if !self.use_residual {
            return self.inner.decode_all(embeddings);
        }

        // Sum all level reconstructions
        let mut result = Array1::zeros(self.inner.input_dim);

        for (level, embedding) in embeddings.iter().enumerate() {
            let decoded = self.inner.decode_level(embedding, level)?;
            result = &result + &decoded;
        }

        Ok(result)
    }

    /// Get number of levels
    pub fn num_levels(&self) -> usize {
        self.inner.num_levels()
    }
}

impl SignalTokenizer for PyramidTokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let embeddings = self.encode_pyramid(signal)?;
        let total_len: usize = embeddings.iter().map(|e| e.len()).sum();
        let mut result = Vec::with_capacity(total_len);
        for emb in embeddings {
            result.extend(emb.iter());
        }
        Ok(Array1::from_vec(result))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let total_dim = self.inner.total_embed_dim();
        if tokens.len() != total_dim {
            return Err(TokenizerError::dim_mismatch(
                total_dim,
                tokens.len(),
                "dimension validation",
            ));
        }

        // Split tokens
        let mut embeddings = Vec::new();
        let mut offset = 0;
        for level in &self.inner.levels {
            let end = offset + level.embed_dim;
            let embedding = Array1::from_vec(
                tokens
                    .iter()
                    .skip(offset)
                    .take(level.embed_dim)
                    .cloned()
                    .collect(),
            );
            embeddings.push(embedding);
            offset = end;
        }

        self.decode_pyramid(&embeddings)
    }

    fn embed_dim(&self) -> usize {
        self.inner.total_embed_dim()
    }

    fn vocab_size(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiscale_basic() {
        let tokenizer = MultiScaleTokenizer::new(64, 16);
        assert_eq!(tokenizer.num_levels(), 3);
        assert_eq!(tokenizer.total_embed_dim(), 48); // 16 * 3

        let signal = Array1::from_vec((0..64).map(|i| (i as f32 * 0.1).sin()).collect());
        let encoded = tokenizer.encode(&signal).unwrap();
        assert_eq!(encoded.len(), 48);

        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded.len(), 64);
    }

    #[test]
    fn test_downsample_average() {
        let tokenizer = MultiScaleTokenizer::new(8, 4);

        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let down = tokenizer.downsample(&signal, 2);

        assert_eq!(down.len(), 4);
        // Average of pairs: (1+2)/2=1.5, (3+4)/2=3.5, ...
        assert!((down[0] - 1.5).abs() < 0.01);
        assert!((down[1] - 3.5).abs() < 0.01);
    }

    #[test]
    fn test_downsample_stride() {
        let tokenizer = MultiScaleTokenizer::new(8, 4).with_pool_method(PoolMethod::Stride);

        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let down = tokenizer.downsample(&signal, 2);

        assert_eq!(down.len(), 4);
        // Every 2nd sample: 1, 3, 5, 7
        assert_eq!(down[0], 1.0);
        assert_eq!(down[1], 3.0);
    }

    #[test]
    fn test_upsample_repeat() {
        let tokenizer = MultiScaleTokenizer::new(8, 4).with_upsample_method(UpsampleMethod::Repeat);

        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let up = tokenizer.upsample(&signal, 2, 8);

        assert_eq!(up.len(), 8);
        assert_eq!(up[0], 1.0);
        assert_eq!(up[1], 1.0);
        assert_eq!(up[2], 2.0);
        assert_eq!(up[3], 2.0);
    }

    #[test]
    fn test_upsample_linear() {
        let tokenizer = MultiScaleTokenizer::new(8, 4).with_upsample_method(UpsampleMethod::Linear);

        let signal = Array1::from_vec(vec![0.0, 2.0]);
        let up = tokenizer.upsample(&signal, 4, 8);

        assert_eq!(up.len(), 8);
        // Linear interp from 0 to 2
        // Position 0 maps to src 0/4 = 0.0 -> signal[0] = 0.0
        // Position 4 maps to src 4/4 = 1.0 -> signal[1] = 2.0
        assert!(up[0].abs() < 0.01);
        // At position 2, src_pos = 0.5, so interp = 0.0*(1-0.5) + 2.0*0.5 = 1.0
        assert!((up[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_encode_level() {
        let tokenizer = MultiScaleTokenizer::new(64, 16);

        let signal = Array1::from_vec((0..64).map(|i| i as f32).collect());

        // Level 0: factor 1 (full res)
        let enc0 = tokenizer.encode_level(&signal, 0).unwrap();
        assert_eq!(enc0.len(), 16);

        // Level 1: factor 2 (half res)
        let enc1 = tokenizer.encode_level(&signal, 1).unwrap();
        assert_eq!(enc1.len(), 16);

        // Level 2: factor 4 (quarter res)
        let enc2 = tokenizer.encode_level(&signal, 2).unwrap();
        assert_eq!(enc2.len(), 16);
    }

    #[test]
    fn test_pyramid_tokenizer() {
        let tokenizer = PyramidTokenizer::new(64, 16, 3);
        assert_eq!(tokenizer.num_levels(), 3);

        let signal = Array1::from_vec((0..64).map(|i| (i as f32 * 0.1).sin()).collect());

        let embeddings = tokenizer.encode_pyramid(&signal).unwrap();
        assert_eq!(embeddings.len(), 3);

        let decoded = tokenizer.decode_pyramid(&embeddings).unwrap();
        assert_eq!(decoded.len(), 64);
    }

    #[test]
    fn test_pyramid_residual() {
        // Residual pyramid should capture progressively finer details
        let tokenizer = PyramidTokenizer::new(32, 8, 3);

        let signal = Array1::from_vec((0..32).map(|i| (i as f32 * 0.2).sin()).collect());

        let embeddings = tokenizer.encode_pyramid(&signal).unwrap();

        // Each level's embedding variance should generally decrease
        // (residuals get smaller as we add more detail)
        let variances: Vec<f32> = embeddings
            .iter()
            .map(|e| {
                let mean = e.sum() / e.len() as f32;
                e.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / e.len() as f32
            })
            .collect();

        // Level 0 should capture most signal variance
        assert!(variances[0] > 0.0);
    }

    #[test]
    fn test_custom_factors() {
        let tokenizer = MultiScaleTokenizer::with_factors(100, 10, &[1, 5, 10, 20]);
        assert_eq!(tokenizer.num_levels(), 4);

        let signal = Array1::from_vec((0..100).map(|i| i as f32).collect());
        let encoded = tokenizer.encode(&signal).unwrap();
        assert_eq!(encoded.len(), 40); // 10 * 4

        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded.len(), 100);
    }
}
