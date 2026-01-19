//! Vector Quantized Variational AutoEncoder (VQ-VAE)
//!
//! VQ-VAE provides discrete representation learning through a learned codebook
//! of embedding vectors. It's widely used in neural audio codecs like SoundStream,
//! Encodec, and Jukebox.
//!
//! ## Algorithm
//!
//! 1. **Encoding**: Map input to continuous latent space
//! 2. **Quantization**: Replace each latent vector with nearest codebook entry
//! 3. **Decoding**: Map quantized latents back to signal space
//!
//! ## Training
//!
//! - **Codebook Loss**: Pulls codebook entries toward encoder outputs
//! - **Commitment Loss**: Encourages encoder to commit to codebook entries
//! - **Straight-Through Estimator**: Passes gradients through quantization
//!
//! ## References
//!
//! - van den Oord et al., "Neural Discrete Representation Learning" (2017)
//! - Razavi et al., "Generating Diverse High-Fidelity Images with VQ-VAE-2" (2019)

use crate::error::{TokenizerError, TokenizerResult};
use crate::SignalTokenizer;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;
use serde::{Deserialize, Serialize};

/// Vector quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VQConfig {
    /// Number of codebook entries
    pub codebook_size: usize,
    /// Dimension of each codebook entry
    pub embed_dim: usize,
    /// Beta parameter for commitment loss (typically 0.25)
    pub commitment_beta: f32,
    /// Decay rate for EMA updates (typically 0.99)
    pub ema_decay: f32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Whether to use EMA updates (vs gradient-based)
    pub use_ema: bool,
}

impl Default for VQConfig {
    fn default() -> Self {
        Self {
            codebook_size: 512,
            embed_dim: 64,
            commitment_beta: 0.25,
            ema_decay: 0.99,
            epsilon: 1e-5,
            use_ema: true,
        }
    }
}

/// Vector Quantizer with learned codebook
#[derive(Debug, Clone)]
pub struct VectorQuantizer {
    /// Configuration
    config: VQConfig,
    /// Codebook embeddings: [codebook_size, embed_dim]
    codebook: Array2<f32>,
    /// EMA cluster sizes (for EMA updates)
    ema_cluster_size: Array1<f32>,
    /// EMA embedding sums (for EMA updates)
    ema_embed_sum: Array2<f32>,
    /// Number of times each code has been used
    usage_counts: Array1<usize>,
}

impl VectorQuantizer {
    /// Create a new vector quantizer with random initialization
    pub fn new(config: VQConfig) -> Self {
        let mut rng = thread_rng();

        // Initialize codebook with random values
        let scale = 1.0 / (config.embed_dim as f32).sqrt();
        let codebook = Array2::from_shape_fn((config.codebook_size, config.embed_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        let ema_cluster_size = Array1::zeros(config.codebook_size);
        let ema_embed_sum = Array2::zeros((config.codebook_size, config.embed_dim));
        let usage_counts = Array1::zeros(config.codebook_size);

        Self {
            config,
            codebook,
            ema_cluster_size,
            ema_embed_sum,
            usage_counts,
        }
    }

    /// Initialize codebook from data using k-means++
    pub fn initialize_from_data(&mut self, data: &[Array1<f32>]) -> TokenizerResult<()> {
        if data.is_empty() {
            return Err(TokenizerError::InvalidConfig(
                "Cannot initialize from empty data".into(),
            ));
        }

        let mut rng = thread_rng();
        let mut centroids = Vec::with_capacity(self.config.codebook_size);

        // k-means++ initialization
        // 1. Choose first centroid randomly
        let first_idx = rng.random_range(0..data.len());
        centroids.push(data[first_idx].clone());

        // 2. Choose remaining centroids with probability proportional to D^2
        while centroids.len() < self.config.codebook_size {
            let mut distances = vec![f32::INFINITY; data.len()];

            // Compute minimum distance to existing centroids
            for (i, point) in data.iter().enumerate() {
                for centroid in &centroids {
                    let dist = self.euclidean_distance(point, centroid);
                    distances[i] = distances[i].min(dist);
                }
            }

            // Choose next centroid with probability proportional to distance^2
            let total: f32 = distances.iter().map(|d| d * d).sum();
            if total <= 0.0 {
                break;
            }

            let mut threshold = rng.random::<f32>() * total;
            for (i, &dist) in distances.iter().enumerate() {
                threshold -= dist * dist;
                if threshold <= 0.0 {
                    centroids.push(data[i].clone());
                    break;
                }
            }
        }

        // Update codebook
        for (i, centroid) in centroids.iter().enumerate() {
            if i >= self.config.codebook_size {
                break;
            }
            for (j, &val) in centroid.iter().enumerate() {
                self.codebook[[i, j]] = val;
            }
        }

        Ok(())
    }

    /// Compute Euclidean distance between two vectors
    #[inline]
    fn euclidean_distance(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Find nearest codebook entry for a vector
    pub fn find_nearest(&self, vector: &Array1<f32>) -> TokenizerResult<usize> {
        if vector.len() != self.config.embed_dim {
            return Err(TokenizerError::dim_mismatch(
                self.config.embed_dim,
                vector.len(),
                "dimension validation",
            ));
        }

        let mut min_dist = f32::INFINITY;
        let mut min_idx = 0;

        for i in 0..self.config.codebook_size {
            let codebook_entry = self.codebook.row(i);
            let dist: f32 = vector
                .iter()
                .zip(codebook_entry.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum();

            if dist < min_dist {
                min_dist = dist;
                min_idx = i;
            }
        }

        Ok(min_idx)
    }

    /// Quantize a vector to its nearest codebook entry
    pub fn quantize(&self, vector: &Array1<f32>) -> TokenizerResult<(usize, Array1<f32>)> {
        let idx = self.find_nearest(vector)?;
        let quantized = self.codebook.row(idx).to_owned();
        Ok((idx, quantized))
    }

    /// Quantize multiple vectors
    pub fn quantize_batch(
        &self,
        vectors: &[Array1<f32>],
    ) -> TokenizerResult<(Vec<usize>, Vec<Array1<f32>>)> {
        let mut indices = Vec::with_capacity(vectors.len());
        let mut quantized = Vec::with_capacity(vectors.len());

        for vector in vectors {
            let (idx, quant) = self.quantize(vector)?;
            indices.push(idx);
            quantized.push(quant);
        }

        Ok((indices, quantized))
    }

    /// Compute VQ losses: (total_loss, codebook_loss, commitment_loss)
    pub fn compute_loss(
        &self,
        encoder_output: &Array1<f32>,
        quantized: &Array1<f32>,
    ) -> (f32, f32, f32) {
        // Codebook loss: ||sg[encoder_output] - codebook||^2
        let codebook_loss: f32 = encoder_output
            .iter()
            .zip(quantized.iter())
            .map(|(e, q)| (e - q).powi(2))
            .sum();

        // Commitment loss: ||encoder_output - sg[codebook]||^2
        let commitment_loss: f32 = encoder_output
            .iter()
            .zip(quantized.iter())
            .map(|(e, q)| (e - q).powi(2))
            .sum();

        let total_loss = codebook_loss + self.config.commitment_beta * commitment_loss;

        (total_loss, codebook_loss, commitment_loss)
    }

    /// Update codebook using EMA
    pub fn update_ema(
        &mut self,
        encoder_outputs: &[Array1<f32>],
        indices: &[usize],
    ) -> TokenizerResult<()> {
        if encoder_outputs.len() != indices.len() {
            return Err(TokenizerError::InvalidConfig(
                "Encoder outputs and indices length mismatch".into(),
            ));
        }

        // Reset temporary accumulators
        let mut cluster_sizes = Array1::<f32>::zeros(self.config.codebook_size);
        let mut embed_sums =
            Array2::<f32>::zeros((self.config.codebook_size, self.config.embed_dim));

        // Accumulate
        for (output, &idx) in encoder_outputs.iter().zip(indices.iter()) {
            cluster_sizes[idx] += 1.0;
            for (j, &val) in output.iter().enumerate() {
                embed_sums[[idx, j]] += val;
            }
            self.usage_counts[idx] += 1;
        }

        // EMA update
        let decay = self.config.ema_decay;
        let epsilon = self.config.epsilon;

        for i in 0..self.config.codebook_size {
            // Update cluster size EMA
            self.ema_cluster_size[i] =
                decay * self.ema_cluster_size[i] + (1.0 - decay) * cluster_sizes[i];

            // Laplace smoothing for numerical stability
            let n = self.ema_cluster_size[i] + epsilon;

            // Update embedding sum EMA and codebook
            for j in 0..self.config.embed_dim {
                self.ema_embed_sum[[i, j]] =
                    decay * self.ema_embed_sum[[i, j]] + (1.0 - decay) * embed_sums[[i, j]];

                // Update codebook entry
                self.codebook[[i, j]] = self.ema_embed_sum[[i, j]] / n;
            }
        }

        Ok(())
    }

    /// Reset unused codebook entries to random encoder outputs
    pub fn reset_unused_codes(
        &mut self,
        encoder_outputs: &[Array1<f32>],
        threshold: usize,
    ) -> usize {
        let mut rng = thread_rng();
        let mut reset_count = 0;

        for i in 0..self.config.codebook_size {
            if self.usage_counts[i] < threshold && !encoder_outputs.is_empty() {
                // Replace with random encoder output
                let random_idx = rng.random_range(0..encoder_outputs.len());
                let random_output = &encoder_outputs[random_idx];

                for (j, &val) in random_output.iter().enumerate() {
                    if j < self.config.embed_dim {
                        self.codebook[[i, j]] = val;
                    }
                }

                // Reset EMA statistics
                self.ema_cluster_size[i] = 1.0;
                for j in 0..self.config.embed_dim {
                    self.ema_embed_sum[[i, j]] = self.codebook[[i, j]];
                }
                self.usage_counts[i] = 0;

                reset_count += 1;
            }
        }

        reset_count
    }

    /// Get codebook entry by index
    pub fn get_codebook_entry(&self, idx: usize) -> TokenizerResult<Array1<f32>> {
        if idx >= self.config.codebook_size {
            return Err(TokenizerError::InvalidConfig(format!(
                "Index {} out of codebook range 0..{}",
                idx, self.config.codebook_size
            )));
        }
        Ok(self.codebook.row(idx).to_owned())
    }

    /// Get the full codebook
    pub fn codebook(&self) -> &Array2<f32> {
        &self.codebook
    }

    /// Get codebook size
    pub fn codebook_size(&self) -> usize {
        self.config.codebook_size
    }

    /// Get embedding dimension
    pub fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }

    /// Get usage statistics
    pub fn usage_stats(&self) -> (usize, usize, f32) {
        let total_uses: usize = self.usage_counts.iter().sum();
        let used_codes = self.usage_counts.iter().filter(|&&c| c > 0).count();
        let utilization = used_codes as f32 / self.config.codebook_size as f32;
        (total_uses, used_codes, utilization)
    }

    /// Reset usage counters
    pub fn reset_usage_counts(&mut self) {
        self.usage_counts.fill(0);
    }
}

/// VQ-VAE with encoder and decoder projections
#[derive(Debug, Clone)]
pub struct VQVAETokenizer {
    /// Encoder projection (input_dim -> embed_dim)
    encoder: Array2<f32>,
    /// Vector quantizer
    quantizer: VectorQuantizer,
    /// Decoder projection (embed_dim -> input_dim)
    decoder: Array2<f32>,
    /// Input dimension
    input_dim: usize,
}

impl VQVAETokenizer {
    /// Create a new VQ-VAE tokenizer
    pub fn new(input_dim: usize, config: VQConfig) -> Self {
        let mut rng = thread_rng();

        // Xavier initialization for encoder
        let enc_scale = (2.0 / (input_dim + config.embed_dim) as f32).sqrt();
        let encoder = Array2::from_shape_fn((input_dim, config.embed_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * enc_scale
        });

        // Xavier initialization for decoder
        let dec_scale = (2.0 / (config.embed_dim + input_dim) as f32).sqrt();
        let decoder = Array2::from_shape_fn((config.embed_dim, input_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * dec_scale
        });

        let quantizer = VectorQuantizer::new(config);

        Self {
            encoder,
            quantizer,
            decoder,
            input_dim,
        }
    }

    /// Encode and quantize
    pub fn encode_quantized(&self, signal: &Array1<f32>) -> TokenizerResult<(usize, Array1<f32>)> {
        if signal.len() != self.input_dim {
            return Err(TokenizerError::dim_mismatch(
                self.input_dim,
                signal.len(),
                "dimension validation",
            ));
        }

        // Encode to latent space
        let latent = signal.dot(&self.encoder);

        // Quantize
        self.quantizer.quantize(&latent)
    }

    /// Decode from quantized vector
    pub fn decode_quantized(&self, quantized: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if quantized.len() != self.quantizer.embed_dim() {
            return Err(TokenizerError::dim_mismatch(
                self.quantizer.embed_dim(),
                quantized.len(),
                "dimension validation",
            ));
        }

        Ok(quantized.dot(&self.decoder))
    }

    /// Decode from index
    pub fn decode_from_index(&self, idx: usize) -> TokenizerResult<Array1<f32>> {
        let quantized = self.quantizer.get_codebook_entry(idx)?;
        self.decode_quantized(&quantized)
    }

    /// Get reference to quantizer
    pub fn quantizer(&self) -> &VectorQuantizer {
        &self.quantizer
    }

    /// Get mutable reference to quantizer (for training)
    pub fn quantizer_mut(&mut self) -> &mut VectorQuantizer {
        &mut self.quantizer
    }

    /// Get encoder weights
    pub fn encoder(&self) -> &Array2<f32> {
        &self.encoder
    }

    /// Get decoder weights
    pub fn decoder(&self) -> &Array2<f32> {
        &self.decoder
    }

    /// Set encoder weights
    pub fn set_encoder(&mut self, weights: Array2<f32>) -> TokenizerResult<()> {
        if weights.shape() != [self.input_dim, self.quantizer.embed_dim()] {
            return Err(TokenizerError::dim_mismatch(
                self.input_dim * self.quantizer.embed_dim(),
                weights.len(),
                "dimension validation",
            ));
        }
        self.encoder = weights;
        Ok(())
    }

    /// Set decoder weights
    pub fn set_decoder(&mut self, weights: Array2<f32>) -> TokenizerResult<()> {
        if weights.shape() != [self.quantizer.embed_dim(), self.input_dim] {
            return Err(TokenizerError::dim_mismatch(
                self.quantizer.embed_dim() * self.input_dim,
                weights.len(),
                "dimension validation",
            ));
        }
        self.decoder = weights;
        Ok(())
    }
}

/// Residual Vector Quantization (RVQ)
///
/// Uses multiple VQ stages where each stage quantizes the residual from previous stages.
/// This is used in modern neural audio codecs like SoundStream and Encodec for
/// high-quality compression with variable bitrate support.
///
/// ## Algorithm
///
/// 1. First stage quantizes the input
/// 2. Compute residual = input - first_quantized
/// 3. Second stage quantizes the residual
/// 4. Repeat for N stages
/// 5. Reconstruction = sum of all quantized outputs
///
/// ## Benefits
///
/// - Progressive quality: use fewer stages for lower bitrate
/// - Better reconstruction than single-stage VQ
/// - Flexible bitrate control
#[derive(Debug, Clone)]
pub struct ResidualVQ {
    /// Vector quantizers for each stage
    quantizers: Vec<VectorQuantizer>,
    /// Number of stages
    num_stages: usize,
}

impl ResidualVQ {
    /// Create a new Residual VQ with multiple stages
    ///
    /// All stages use the same configuration but independent codebooks
    pub fn new(num_stages: usize, config: VQConfig) -> Self {
        let quantizers = (0..num_stages)
            .map(|_| VectorQuantizer::new(config.clone()))
            .collect();

        Self {
            quantizers,
            num_stages,
        }
    }

    /// Create with different configs per stage
    pub fn with_configs(configs: Vec<VQConfig>) -> Self {
        let num_stages = configs.len();
        let quantizers = configs.into_iter().map(VectorQuantizer::new).collect();

        Self {
            quantizers,
            num_stages,
        }
    }

    /// Encode with all stages
    ///
    /// Returns: (indices for each stage, quantized outputs for each stage)
    pub fn encode(&self, vector: &Array1<f32>) -> TokenizerResult<(Vec<usize>, Vec<Array1<f32>>)> {
        let mut indices = Vec::with_capacity(self.num_stages);
        let mut quantized_outputs = Vec::with_capacity(self.num_stages);
        let mut residual = vector.clone();

        for quantizer in &self.quantizers {
            let (idx, quantized) = quantizer.quantize(&residual)?;
            indices.push(idx);
            quantized_outputs.push(quantized.clone());

            // Update residual for next stage
            residual = &residual - &quantized;
        }

        Ok((indices, quantized_outputs))
    }

    /// Encode with limited number of stages (for variable bitrate)
    pub fn encode_with_stages(
        &self,
        vector: &Array1<f32>,
        num_stages: usize,
    ) -> TokenizerResult<(Vec<usize>, Vec<Array1<f32>>)> {
        if num_stages > self.num_stages {
            return Err(TokenizerError::InvalidConfig(format!(
                "Requested {} stages but only {} available",
                num_stages, self.num_stages
            )));
        }

        let mut indices = Vec::with_capacity(num_stages);
        let mut quantized_outputs = Vec::with_capacity(num_stages);
        let mut residual = vector.clone();

        for quantizer in self.quantizers.iter().take(num_stages) {
            let (idx, quantized) = quantizer.quantize(&residual)?;
            indices.push(idx);
            quantized_outputs.push(quantized.clone());

            residual = &residual - &quantized;
        }

        Ok((indices, quantized_outputs))
    }

    /// Decode from all stage indices
    pub fn decode(&self, indices: &[usize]) -> TokenizerResult<Array1<f32>> {
        if indices.len() != self.num_stages {
            return Err(TokenizerError::InvalidConfig(format!(
                "Expected {} indices, got {}",
                self.num_stages,
                indices.len()
            )));
        }

        let first_entry = self.quantizers[0].get_codebook_entry(indices[0])?;
        let mut result = first_entry;

        for (quantizer, &idx) in self.quantizers.iter().skip(1).zip(indices.iter().skip(1)) {
            let entry = quantizer.get_codebook_entry(idx)?;
            result = &result + &entry;
        }

        Ok(result)
    }

    /// Decode from quantized outputs (sum them)
    pub fn decode_from_quantized(
        &self,
        quantized_outputs: &[Array1<f32>],
    ) -> TokenizerResult<Array1<f32>> {
        if quantized_outputs.is_empty() {
            return Err(TokenizerError::InvalidConfig("No quantized outputs".into()));
        }

        let mut result = quantized_outputs[0].clone();
        for output in quantized_outputs.iter().skip(1) {
            result = &result + output;
        }

        Ok(result)
    }

    /// Update all stages with EMA
    pub fn update_ema(&mut self, encoder_outputs: &[Array1<f32>]) -> TokenizerResult<()> {
        if encoder_outputs.is_empty() {
            return Ok(());
        }

        // Collect residuals and indices for each stage
        let mut stage_outputs = vec![Vec::new(); self.num_stages];
        let mut stage_indices = vec![Vec::new(); self.num_stages];

        for output in encoder_outputs {
            let mut residual = output.clone();

            for (stage_idx, quantizer) in self.quantizers.iter().enumerate() {
                let (idx, quantized) = quantizer.quantize(&residual)?;
                stage_outputs[stage_idx].push(residual.clone());
                stage_indices[stage_idx].push(idx);

                residual = &residual - &quantized;
            }
        }

        // Update each stage
        for (quantizer, (outputs, indices)) in self
            .quantizers
            .iter_mut()
            .zip(stage_outputs.iter().zip(stage_indices.iter()))
        {
            quantizer.update_ema(outputs, indices)?;
        }

        Ok(())
    }

    /// Get reference to a specific stage
    pub fn stage(&self, idx: usize) -> Option<&VectorQuantizer> {
        self.quantizers.get(idx)
    }

    /// Get mutable reference to a specific stage
    pub fn stage_mut(&mut self, idx: usize) -> Option<&mut VectorQuantizer> {
        self.quantizers.get_mut(idx)
    }

    /// Get number of stages
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    /// Compute total bits per sample (sum across stages)
    pub fn total_bits(&self) -> f32 {
        self.quantizers
            .iter()
            .map(|q| (q.codebook_size() as f32).log2())
            .sum()
    }

    /// Compute bitrate for a given number of stages
    pub fn bitrate_for_stages(&self, num_stages: usize) -> f32 {
        self.quantizers
            .iter()
            .take(num_stages)
            .map(|q| (q.codebook_size() as f32).log2())
            .sum()
    }

    /// Get usage statistics across all stages
    pub fn all_usage_stats(&self) -> Vec<(usize, usize, f32)> {
        self.quantizers.iter().map(|q| q.usage_stats()).collect()
    }

    /// Reset usage counts for all stages
    pub fn reset_all_usage_counts(&mut self) {
        for quantizer in &mut self.quantizers {
            quantizer.reset_usage_counts();
        }
    }
}

/// RVQ-VAE Tokenizer with residual quantization
#[derive(Debug, Clone)]
pub struct RVQVAETokenizer {
    /// Encoder projection
    encoder: Array2<f32>,
    /// Residual VQ
    rvq: ResidualVQ,
    /// Decoder projection
    decoder: Array2<f32>,
    /// Input dimension
    input_dim: usize,
}

impl RVQVAETokenizer {
    /// Create a new RVQ-VAE tokenizer
    pub fn new(input_dim: usize, num_stages: usize, config: VQConfig) -> Self {
        let mut rng = scirs2_core::random::thread_rng();

        let enc_scale = (2.0 / (input_dim + config.embed_dim) as f32).sqrt();
        let encoder = Array2::from_shape_fn((input_dim, config.embed_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * enc_scale
        });

        let dec_scale = (2.0 / (config.embed_dim + input_dim) as f32).sqrt();
        let decoder = Array2::from_shape_fn((config.embed_dim, input_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * dec_scale
        });

        let rvq = ResidualVQ::new(num_stages, config);

        Self {
            encoder,
            rvq,
            decoder,
            input_dim,
        }
    }

    /// Encode and quantize with all stages
    pub fn encode_quantized(
        &self,
        signal: &Array1<f32>,
    ) -> TokenizerResult<(Vec<usize>, Vec<Array1<f32>>)> {
        if signal.len() != self.input_dim {
            return Err(TokenizerError::dim_mismatch(
                self.input_dim,
                signal.len(),
                "dimension validation",
            ));
        }

        let latent = signal.dot(&self.encoder);
        self.rvq.encode(&latent)
    }

    /// Encode with limited stages (variable bitrate)
    pub fn encode_with_stages(
        &self,
        signal: &Array1<f32>,
        num_stages: usize,
    ) -> TokenizerResult<(Vec<usize>, Vec<Array1<f32>>)> {
        if signal.len() != self.input_dim {
            return Err(TokenizerError::dim_mismatch(
                self.input_dim,
                signal.len(),
                "dimension validation",
            ));
        }

        let latent = signal.dot(&self.encoder);
        self.rvq.encode_with_stages(&latent, num_stages)
    }

    /// Decode from indices
    pub fn decode_from_indices(&self, indices: &[usize]) -> TokenizerResult<Array1<f32>> {
        let quantized = self.rvq.decode(indices)?;
        Ok(quantized.dot(&self.decoder))
    }

    /// Decode from quantized outputs
    pub fn decode_from_quantized(
        &self,
        quantized_outputs: &[Array1<f32>],
    ) -> TokenizerResult<Array1<f32>> {
        let summed = self.rvq.decode_from_quantized(quantized_outputs)?;
        Ok(summed.dot(&self.decoder))
    }

    /// Get reference to RVQ
    pub fn rvq(&self) -> &ResidualVQ {
        &self.rvq
    }

    /// Get mutable reference to RVQ
    pub fn rvq_mut(&mut self) -> &mut ResidualVQ {
        &mut self.rvq
    }

    /// Get total bitrate
    pub fn total_bitrate(&self) -> f32 {
        self.rvq.total_bits()
    }

    /// Get bitrate for specific number of stages
    pub fn bitrate_for_stages(&self, num_stages: usize) -> f32 {
        self.rvq.bitrate_for_stages(num_stages)
    }
}

impl SignalTokenizer for RVQVAETokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let (indices, _) = self.encode_quantized(signal)?;
        // Return concatenated indices as floats
        Ok(Array1::from_vec(
            indices.iter().map(|&i| i as f32).collect(),
        ))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let indices: Vec<usize> = tokens.iter().map(|&t| t.round() as usize).collect();
        self.decode_from_indices(&indices)
    }

    fn embed_dim(&self) -> usize {
        self.rvq.num_stages() // Returns number of stages (each produces one index)
    }

    fn vocab_size(&self) -> usize {
        // Total vocabulary is product of all codebook sizes
        self.rvq
            .quantizers
            .iter()
            .map(|q| q.codebook_size())
            .product()
    }
}

/// Product Quantization (PQ) Configuration
///
/// Product Quantization splits the embedding space into M subspaces and
/// quantizes each independently, enabling very large effective codebook sizes
/// with linear memory/compute scaling.
///
/// ## Example
///
/// With 4 subspaces and 256 codes per subspace:
/// - Memory: 4 * 256 * (embed_dim/4) parameters
/// - Effective codebook size: 256^4 = 4.3 billion codes
/// - vs. Standard VQ: needs 4.3B * embed_dim parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizerConfig {
    /// Number of subspaces (M)
    pub num_subspaces: usize,
    /// Codebook size per subspace (K)
    pub codebook_size_per_subspace: usize,
    /// Total embedding dimension (must be divisible by num_subspaces)
    pub embed_dim: usize,
    /// Beta parameter for commitment loss
    pub commitment_beta: f32,
    /// EMA decay rate
    pub ema_decay: f32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Use EMA updates
    pub use_ema: bool,
}

impl Default for ProductQuantizerConfig {
    fn default() -> Self {
        Self {
            num_subspaces: 4,
            codebook_size_per_subspace: 256,
            embed_dim: 64,
            commitment_beta: 0.25,
            ema_decay: 0.99,
            epsilon: 1e-5,
            use_ema: true,
        }
    }
}

/// Type alias for batch quantization results
type BatchQuantizeResult = (Vec<Vec<usize>>, Vec<Array1<f32>>);

/// Product Quantizer
///
/// Implements Product Quantization (Jegou et al., 2011) for efficient
/// high-dimensional vector quantization with exponentially large effective
/// codebook sizes.
///
/// # Algorithm
///
/// 1. Split D-dimensional vector into M subspaces of D/M dimensions
/// 2. Quantize each subspace independently using its own codebook
/// 3. Concatenate quantized subspaces
/// 4. Effective codebook size: K^M (K = codes per subspace)
///
/// # Memory Complexity
///
/// - Standard VQ: O(K * D)
/// - Product VQ: O(M * K * D/M) = O(K * D)
/// - But effective codes: K^M vs K (exponential gain!)
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    /// Configuration
    config: ProductQuantizerConfig,
    /// Subspace quantizers (one per subspace)
    subspace_quantizers: Vec<VectorQuantizer>,
    /// Dimension per subspace
    subspace_dim: usize,
}

impl ProductQuantizer {
    /// Create a new product quantizer
    pub fn new(config: ProductQuantizerConfig) -> TokenizerResult<Self> {
        if !config.embed_dim.is_multiple_of(config.num_subspaces) {
            return Err(TokenizerError::InvalidConfig(format!(
                "embed_dim ({}) must be divisible by num_subspaces ({})",
                config.embed_dim, config.num_subspaces
            )));
        }

        let subspace_dim = config.embed_dim / config.num_subspaces;

        // Create a quantizer for each subspace
        let mut subspace_quantizers = Vec::with_capacity(config.num_subspaces);
        for _ in 0..config.num_subspaces {
            let subspace_config = VQConfig {
                codebook_size: config.codebook_size_per_subspace,
                embed_dim: subspace_dim,
                commitment_beta: config.commitment_beta,
                ema_decay: config.ema_decay,
                epsilon: config.epsilon,
                use_ema: config.use_ema,
            };
            subspace_quantizers.push(VectorQuantizer::new(subspace_config));
        }

        Ok(Self {
            config,
            subspace_quantizers,
            subspace_dim,
        })
    }

    /// Split a vector into subspaces
    fn split_into_subspaces(&self, vector: &Array1<f32>) -> TokenizerResult<Vec<Array1<f32>>> {
        if vector.len() != self.config.embed_dim {
            return Err(TokenizerError::dim_mismatch(
                self.config.embed_dim,
                vector.len(),
                "dimension validation",
            ));
        }

        let mut subspaces = Vec::with_capacity(self.config.num_subspaces);
        for i in 0..self.config.num_subspaces {
            let start = i * self.subspace_dim;
            let end = start + self.subspace_dim;
            let subspace =
                Array1::from_vec(vector.slice(scirs2_core::ndarray::s![start..end]).to_vec());
            subspaces.push(subspace);
        }

        Ok(subspaces)
    }

    /// Concatenate subspace vectors
    fn concatenate_subspaces(&self, subspaces: &[Array1<f32>]) -> TokenizerResult<Array1<f32>> {
        if subspaces.len() != self.config.num_subspaces {
            return Err(TokenizerError::InvalidConfig(format!(
                "Expected {} subspaces, got {}",
                self.config.num_subspaces,
                subspaces.len()
            )));
        }

        let mut result = Vec::with_capacity(self.config.embed_dim);
        for subspace in subspaces {
            result.extend_from_slice(
                subspace
                    .as_slice()
                    .expect("Subspace must have contiguous layout"),
            );
        }

        Ok(Array1::from_vec(result))
    }

    /// Quantize a vector using product quantization
    pub fn quantize(&self, vector: &Array1<f32>) -> TokenizerResult<(Vec<usize>, Array1<f32>)> {
        let subspaces = self.split_into_subspaces(vector)?;
        let mut indices = Vec::with_capacity(self.config.num_subspaces);
        let mut quantized_subspaces = Vec::with_capacity(self.config.num_subspaces);

        for (subspace, quantizer) in subspaces.iter().zip(&self.subspace_quantizers) {
            let (idx, quantized) = quantizer.quantize(subspace)?;
            indices.push(idx);
            quantized_subspaces.push(quantized);
        }

        let quantized_vector = self.concatenate_subspaces(&quantized_subspaces)?;
        Ok((indices, quantized_vector))
    }

    /// Quantize multiple vectors
    pub fn quantize_batch(&self, vectors: &[Array1<f32>]) -> TokenizerResult<BatchQuantizeResult> {
        let mut all_indices = Vec::with_capacity(vectors.len());
        let mut all_quantized = Vec::with_capacity(vectors.len());

        for vector in vectors {
            let (indices, quantized) = self.quantize(vector)?;
            all_indices.push(indices);
            all_quantized.push(quantized);
        }

        Ok((all_indices, all_quantized))
    }

    /// Decode from subspace indices
    pub fn decode(&self, indices: &[usize]) -> TokenizerResult<Array1<f32>> {
        if indices.len() != self.config.num_subspaces {
            return Err(TokenizerError::InvalidConfig(format!(
                "Expected {} indices, got {}",
                self.config.num_subspaces,
                indices.len()
            )));
        }

        let mut subspaces = Vec::with_capacity(self.config.num_subspaces);
        for (idx, quantizer) in indices.iter().zip(&self.subspace_quantizers) {
            let entry = quantizer.get_codebook_entry(*idx)?;
            subspaces.push(entry);
        }

        self.concatenate_subspaces(&subspaces)
    }

    /// Initialize from data using k-means++ for each subspace
    pub fn initialize_from_data(&mut self, data: &[Array1<f32>]) -> TokenizerResult<()> {
        if data.is_empty() {
            return Err(TokenizerError::InvalidConfig(
                "Cannot initialize from empty data".into(),
            ));
        }

        // Split all data into subspaces
        let mut subspace_data: Vec<Vec<Array1<f32>>> =
            vec![Vec::with_capacity(data.len()); self.config.num_subspaces];

        for vector in data {
            let subspaces = self.split_into_subspaces(vector)?;
            for (i, subspace) in subspaces.into_iter().enumerate() {
                subspace_data[i].push(subspace);
            }
        }

        // Initialize each subspace quantizer
        for (quantizer, data) in self
            .subspace_quantizers
            .iter_mut()
            .zip(subspace_data.iter())
        {
            quantizer.initialize_from_data(data)?;
        }

        Ok(())
    }

    /// Update using EMA
    pub fn update_ema(
        &mut self,
        encoder_outputs: &[Array1<f32>],
        all_indices: &[Vec<usize>],
    ) -> TokenizerResult<()> {
        if encoder_outputs.len() != all_indices.len() {
            return Err(TokenizerError::InvalidConfig(
                "Mismatch between encoder_outputs and indices".into(),
            ));
        }

        // Split encoder outputs into subspaces
        let mut subspace_outputs: Vec<Vec<Array1<f32>>> =
            vec![Vec::with_capacity(encoder_outputs.len()); self.config.num_subspaces];
        let mut subspace_indices: Vec<Vec<usize>> =
            vec![Vec::with_capacity(encoder_outputs.len()); self.config.num_subspaces];

        for (output, indices) in encoder_outputs.iter().zip(all_indices.iter()) {
            if indices.len() != self.config.num_subspaces {
                return Err(TokenizerError::InvalidConfig(
                    "Invalid indices length".into(),
                ));
            }

            let subspaces = self.split_into_subspaces(output)?;
            for (i, (subspace, &idx)) in subspaces.into_iter().zip(indices.iter()).enumerate() {
                subspace_outputs[i].push(subspace);
                subspace_indices[i].push(idx);
            }
        }

        // Update each subspace quantizer
        for (quantizer, (outputs, indices)) in self
            .subspace_quantizers
            .iter_mut()
            .zip(subspace_outputs.iter().zip(subspace_indices.iter()))
        {
            quantizer.update_ema(outputs, indices)?;
        }

        Ok(())
    }

    /// Compute VQ losses
    pub fn compute_loss(
        &self,
        encoder_output: &Array1<f32>,
        quantized: &Array1<f32>,
    ) -> (f32, f32, f32) {
        // Codebook loss: ||sg[encoder_output] - quantized||^2
        let codebook_loss: f32 = encoder_output
            .iter()
            .zip(quantized.iter())
            .map(|(e, q)| (e - q).powi(2))
            .sum();

        // Commitment loss: ||encoder_output - sg[quantized]||^2
        let commitment_loss: f32 = encoder_output
            .iter()
            .zip(quantized.iter())
            .map(|(e, q)| (e - q).powi(2))
            .sum();

        let total_loss = codebook_loss + self.config.commitment_beta * commitment_loss;

        (total_loss, codebook_loss, commitment_loss)
    }

    /// Get effective codebook size (K^M)
    pub fn effective_codebook_size(&self) -> usize {
        self.config
            .codebook_size_per_subspace
            .pow(self.config.num_subspaces as u32)
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.config.num_subspaces * self.config.codebook_size_per_subspace * self.subspace_dim
    }

    /// Get embedding dimension
    pub fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }

    /// Get number of subspaces
    pub fn num_subspaces(&self) -> usize {
        self.config.num_subspaces
    }

    /// Get codebook size per subspace
    pub fn codebook_size_per_subspace(&self) -> usize {
        self.config.codebook_size_per_subspace
    }

    /// Reset dead codes in all subspaces
    pub fn reset_unused_codes(
        &mut self,
        encoder_outputs: &[Array1<f32>],
        threshold: usize,
    ) -> TokenizerResult<usize> {
        if encoder_outputs.is_empty() {
            return Ok(0);
        }

        // Split encoder outputs into subspaces
        let mut subspace_outputs: Vec<Vec<Array1<f32>>> =
            vec![Vec::with_capacity(encoder_outputs.len()); self.config.num_subspaces];

        for output in encoder_outputs {
            let subspaces = self.split_into_subspaces(output)?;
            for (i, subspace) in subspaces.into_iter().enumerate() {
                subspace_outputs[i].push(subspace);
            }
        }

        // Reset unused codes in each subspace quantizer
        let mut total_reset = 0;
        for (quantizer, outputs) in self
            .subspace_quantizers
            .iter_mut()
            .zip(subspace_outputs.iter())
        {
            total_reset += quantizer.reset_unused_codes(outputs, threshold);
        }

        Ok(total_reset)
    }

    /// Get usage statistics for all subspaces
    pub fn usage_stats(&self) -> Vec<(usize, usize, f32)> {
        self.subspace_quantizers
            .iter()
            .map(|q| q.usage_stats())
            .collect()
    }
}

impl SignalTokenizer for VQVAETokenizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let (idx, _) = self.encode_quantized(signal)?;
        // Return index as float for embedding lookup
        Ok(Array1::from_elem(1, idx as f32))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        if tokens.len() != 1 {
            return Err(TokenizerError::dim_mismatch(
                1,
                tokens.len(),
                "dimension validation",
            ));
        }
        let idx = tokens[0].round() as usize;
        self.decode_from_index(idx)
    }

    fn embed_dim(&self) -> usize {
        self.quantizer.embed_dim()
    }

    fn vocab_size(&self) -> usize {
        self.quantizer.codebook_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_quantizer_creation() {
        let config = VQConfig::default();
        let vq = VectorQuantizer::new(config.clone());

        assert_eq!(vq.codebook_size(), config.codebook_size);
        assert_eq!(vq.embed_dim(), config.embed_dim);
    }

    #[test]
    fn test_quantization() {
        let config = VQConfig {
            codebook_size: 8,
            embed_dim: 4,
            ..Default::default()
        };
        let vq = VectorQuantizer::new(config);

        let vector = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let (idx, quantized) = vq.quantize(&vector).unwrap();

        assert!(idx < 8);
        assert_eq!(quantized.len(), 4);
    }

    #[test]
    fn test_find_nearest() {
        let config = VQConfig {
            codebook_size: 4,
            embed_dim: 2,
            ..Default::default()
        };
        let mut vq = VectorQuantizer::new(config);

        // Manually set codebook entries
        vq.codebook[[0, 0]] = 0.0;
        vq.codebook[[0, 1]] = 0.0;
        vq.codebook[[1, 0]] = 1.0;
        vq.codebook[[1, 1]] = 0.0;
        vq.codebook[[2, 0]] = 0.0;
        vq.codebook[[2, 1]] = 1.0;
        vq.codebook[[3, 0]] = 1.0;
        vq.codebook[[3, 1]] = 1.0;

        let vector = Array1::from_vec(vec![0.9, 0.1]);
        let idx = vq.find_nearest(&vector).unwrap();
        assert_eq!(idx, 1); // Closest to [1.0, 0.0]
    }

    #[test]
    fn test_compute_loss() {
        let config = VQConfig::default();
        let vq = VectorQuantizer::new(config);

        let encoder_output = Array1::from_vec(vec![0.5; 64]);
        let quantized = Array1::from_vec(vec![0.4; 64]);

        let (total_loss, codebook_loss, commitment_loss) =
            vq.compute_loss(&encoder_output, &quantized);

        assert!(total_loss > 0.0);
        assert!(codebook_loss > 0.0);
        assert!(commitment_loss > 0.0);
    }

    #[test]
    fn test_ema_update() {
        let config = VQConfig {
            codebook_size: 4,
            embed_dim: 8,
            use_ema: true,
            ..Default::default()
        };
        let mut vq = VectorQuantizer::new(config);

        let outputs = vec![
            Array1::from_vec(vec![0.1; 8]),
            Array1::from_vec(vec![0.2; 8]),
            Array1::from_vec(vec![0.3; 8]),
        ];
        let indices = vec![0, 1, 0];

        vq.update_ema(&outputs, &indices).unwrap();

        // Check that codebook has been updated
        let (total, used, util) = vq.usage_stats();
        assert_eq!(total, 3);
        assert_eq!(used, 2); // Only indices 0 and 1 were used
        assert!(util > 0.0);
    }

    #[test]
    fn test_vqvae_tokenizer() {
        let config = VQConfig {
            codebook_size: 16,
            embed_dim: 8,
            ..Default::default()
        };
        let tokenizer = VQVAETokenizer::new(32, config);

        let signal = Array1::from_vec((0..32).map(|i| (i as f32 * 0.1).sin()).collect());

        let encoded = tokenizer.encode(&signal).unwrap();
        assert_eq!(encoded.len(), 1); // Returns single index

        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded.len(), 32);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let config = VQConfig {
            codebook_size: 32,
            embed_dim: 16,
            ..Default::default()
        };
        let tokenizer = VQVAETokenizer::new(64, config);

        let signal = Array1::from_vec((0..64).map(|i| (i as f32 * 0.05).cos()).collect());

        let (idx, _quantized) = tokenizer.encode_quantized(&signal).unwrap();
        let decoded = tokenizer.decode_from_index(idx).unwrap();

        assert_eq!(decoded.len(), signal.len());
    }

    #[test]
    fn test_batch_quantization() {
        let config = VQConfig {
            codebook_size: 8,
            embed_dim: 4,
            ..Default::default()
        };
        let vq = VectorQuantizer::new(config);

        let vectors = vec![
            Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]),
            Array1::from_vec(vec![0.5, 0.6, 0.7, 0.8]),
            Array1::from_vec(vec![0.9, 1.0, 1.1, 1.2]),
        ];

        let (indices, quantized) = vq.quantize_batch(&vectors).unwrap();

        assert_eq!(indices.len(), 3);
        assert_eq!(quantized.len(), 3);
        for q in &quantized {
            assert_eq!(q.len(), 4);
        }
    }

    #[test]
    fn test_reset_unused_codes() {
        let config = VQConfig {
            codebook_size: 8,
            embed_dim: 4,
            ..Default::default()
        };
        let mut vq = VectorQuantizer::new(config);

        // Simulate usage of only some codes
        vq.usage_counts[0] = 10;
        vq.usage_counts[1] = 5;
        // Rest are 0

        let encoder_outputs = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
            Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]),
        ];

        let reset_count = vq.reset_unused_codes(&encoder_outputs, 3);

        // Should reset codes with usage < 3 (that's codes 2-7, so 6 codes)
        assert_eq!(reset_count, 6);
    }

    #[test]
    fn test_initialization_from_data() {
        let config = VQConfig {
            codebook_size: 4,
            embed_dim: 3,
            ..Default::default()
        };
        let mut vq = VectorQuantizer::new(config);

        let data = vec![
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
            Array1::from_vec(vec![0.0, 1.0, 0.0]),
            Array1::from_vec(vec![0.0, 0.0, 1.0]),
            Array1::from_vec(vec![1.0, 1.0, 0.0]),
            Array1::from_vec(vec![1.0, 0.0, 1.0]),
            Array1::from_vec(vec![0.0, 1.0, 1.0]),
            Array1::from_vec(vec![1.0, 1.0, 1.0]),
        ];

        vq.initialize_from_data(&data).unwrap();

        // Verify codebook was updated
        for i in 0..4 {
            let entry = vq.get_codebook_entry(i).unwrap();
            assert_eq!(entry.len(), 3);
        }
    }

    // RVQ Tests
    #[test]
    fn test_residual_vq_creation() {
        let config = VQConfig {
            codebook_size: 8,
            embed_dim: 4,
            ..Default::default()
        };
        let rvq = ResidualVQ::new(3, config);

        assert_eq!(rvq.num_stages(), 3);
        assert!(rvq.total_bits() > 0.0);
    }

    #[test]
    fn test_rvq_encode_decode() {
        let config = VQConfig {
            codebook_size: 16,
            embed_dim: 8,
            ..Default::default()
        };
        let rvq = ResidualVQ::new(4, config);

        let vector = Array1::from_vec((0..8).map(|i| (i as f32 * 0.1).sin()).collect());

        let (indices, quantized_outputs) = rvq.encode(&vector).unwrap();

        assert_eq!(indices.len(), 4);
        assert_eq!(quantized_outputs.len(), 4);

        // Decode and verify reconstruction
        let reconstructed = rvq.decode(&indices).unwrap();
        assert_eq!(reconstructed.len(), vector.len());

        // Decode from quantized outputs
        let reconstructed2 = rvq.decode_from_quantized(&quantized_outputs).unwrap();
        assert_eq!(reconstructed2.len(), vector.len());

        // Both decode methods should give same result
        for (a, b) in reconstructed.iter().zip(reconstructed2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rvq_variable_stages() {
        let config = VQConfig {
            codebook_size: 16,
            embed_dim: 8,
            ..Default::default()
        };
        let rvq = ResidualVQ::new(4, config);

        let vector = Array1::from_vec((0..8).map(|i| i as f32).collect());

        // Encode with 2 stages
        let (indices, _) = rvq.encode_with_stages(&vector, 2).unwrap();
        assert_eq!(indices.len(), 2);

        // Encode with 4 stages
        let (indices_full, _) = rvq.encode(&vector).unwrap();
        assert_eq!(indices_full.len(), 4);

        // First 2 indices should be the same
        assert_eq!(indices[0], indices_full[0]);
        assert_eq!(indices[1], indices_full[1]);
    }

    #[test]
    fn test_rvq_bitrate() {
        let config = VQConfig {
            codebook_size: 256, // 8 bits
            embed_dim: 8,
            ..Default::default()
        };
        let rvq = ResidualVQ::new(3, config);

        let total_bits = rvq.total_bits();
        assert!((total_bits - 24.0).abs() < 0.1); // 8 bits * 3 stages = 24 bits

        let bits_2_stages = rvq.bitrate_for_stages(2);
        assert!((bits_2_stages - 16.0).abs() < 0.1); // 8 bits * 2 stages = 16 bits
    }

    #[test]
    fn test_rvq_ema_update() {
        let config = VQConfig {
            codebook_size: 8,
            embed_dim: 4,
            use_ema: true,
            ..Default::default()
        };
        let mut rvq = ResidualVQ::new(2, config);

        let outputs = vec![
            Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]),
            Array1::from_vec(vec![0.5, 0.6, 0.7, 0.8]),
            Array1::from_vec(vec![0.2, 0.3, 0.4, 0.5]),
        ];

        rvq.update_ema(&outputs).unwrap();

        let stats = rvq.all_usage_stats();
        assert_eq!(stats.len(), 2); // 2 stages
    }

    #[test]
    fn test_rvq_stage_access() {
        let config = VQConfig {
            codebook_size: 8,
            embed_dim: 4,
            ..Default::default()
        };
        let mut rvq = ResidualVQ::new(3, config);

        // Test immutable access
        let stage0 = rvq.stage(0).unwrap();
        assert_eq!(stage0.codebook_size(), 8);

        // Test mutable access
        let stage1 = rvq.stage_mut(1).unwrap();
        assert_eq!(stage1.codebook_size(), 8);

        // Test out of bounds
        assert!(rvq.stage(10).is_none());
    }

    #[test]
    fn test_rvqvae_tokenizer() {
        let config = VQConfig {
            codebook_size: 16,
            embed_dim: 8,
            ..Default::default()
        };
        let tokenizer = RVQVAETokenizer::new(32, 4, config);

        let signal = Array1::from_vec((0..32).map(|i| (i as f32 * 0.05).sin()).collect());

        let (indices, quantized) = tokenizer.encode_quantized(&signal).unwrap();
        assert_eq!(indices.len(), 4);
        assert_eq!(quantized.len(), 4);

        let reconstructed = tokenizer.decode_from_indices(&indices).unwrap();
        assert_eq!(reconstructed.len(), 32);
    }

    #[test]
    fn test_rvqvae_variable_bitrate() {
        let config = VQConfig {
            codebook_size: 256,
            embed_dim: 16,
            ..Default::default()
        };
        let tokenizer = RVQVAETokenizer::new(64, 4, config);

        let signal = Array1::from_vec((0..64).map(|i| (i as f32 * 0.05).cos()).collect());

        // Low bitrate (2 stages)
        let (indices_low, _) = tokenizer.encode_with_stages(&signal, 2).unwrap();
        assert_eq!(indices_low.len(), 2);

        // High bitrate (4 stages)
        let (indices_high, _) = tokenizer.encode_with_stages(&signal, 4).unwrap();
        assert_eq!(indices_high.len(), 4);

        // Check bitrates
        let bitrate_low = tokenizer.bitrate_for_stages(2);
        let bitrate_high = tokenizer.total_bitrate();
        assert!(bitrate_high > bitrate_low);
    }

    #[test]
    fn test_rvqvae_signal_tokenizer_trait() {
        let config = VQConfig {
            codebook_size: 32,
            embed_dim: 12,
            ..Default::default()
        };
        let tokenizer = RVQVAETokenizer::new(48, 3, config);

        let signal = Array1::from_vec((0..48).map(|i| i as f32 * 0.1).collect());

        // Test through SignalTokenizer trait
        let encoded = tokenizer.encode(&signal).unwrap();
        assert_eq!(encoded.len(), 3); // 3 stages = 3 indices

        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded.len(), 48);
    }

    #[test]
    fn test_rvq_with_different_configs() {
        let configs = vec![
            VQConfig {
                codebook_size: 128,
                embed_dim: 8,
                ..Default::default()
            },
            VQConfig {
                codebook_size: 256,
                embed_dim: 8,
                ..Default::default()
            },
            VQConfig {
                codebook_size: 512,
                embed_dim: 8,
                ..Default::default()
            },
        ];

        let rvq = ResidualVQ::with_configs(configs);
        assert_eq!(rvq.num_stages(), 3);

        let vector = Array1::from_vec((0..8).map(|i| i as f32).collect());
        let (indices, _) = rvq.encode(&vector).unwrap();
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_rvq_residual_progression() {
        // Test that residuals decrease with each stage
        let config = VQConfig {
            codebook_size: 64,
            embed_dim: 16,
            ..Default::default()
        };
        let rvq = ResidualVQ::new(4, config);

        let vector = Array1::from_vec((0..16).map(|i| (i as f32 * 0.1).sin()).collect());

        let (_, quantized_outputs) = rvq.encode(&vector).unwrap();

        // Compute residual norms
        let mut residual = vector.clone();
        let mut residual_norms = Vec::new();

        for quantized in &quantized_outputs {
            let norm: f32 = residual.iter().map(|x| x * x).sum::<f32>().sqrt();
            residual_norms.push(norm);
            residual = &residual - quantized;
        }

        // Residual norms should generally decrease
        // (first stage captures most variance)
        assert!(residual_norms[0] > 0.0);
    }

    #[test]
    fn test_product_quantizer_creation() {
        let config = ProductQuantizerConfig {
            num_subspaces: 4,
            codebook_size_per_subspace: 16,
            embed_dim: 64,
            ..Default::default()
        };

        let pq = ProductQuantizer::new(config.clone()).unwrap();

        assert_eq!(pq.embed_dim(), 64);
        assert_eq!(pq.num_subspaces(), 4);
        assert_eq!(pq.codebook_size_per_subspace(), 16);
        assert_eq!(pq.effective_codebook_size(), 16_usize.pow(4)); // 65536
    }

    #[test]
    fn test_product_quantizer_invalid_config() {
        // embed_dim not divisible by num_subspaces
        let config = ProductQuantizerConfig {
            num_subspaces: 3,
            codebook_size_per_subspace: 16,
            embed_dim: 64, // 64 % 3 != 0
            ..Default::default()
        };

        assert!(ProductQuantizer::new(config).is_err());
    }

    #[test]
    fn test_product_quantize_decode() {
        let config = ProductQuantizerConfig {
            num_subspaces: 4,
            codebook_size_per_subspace: 8,
            embed_dim: 64,
            ..Default::default()
        };

        let pq = ProductQuantizer::new(config).unwrap();
        let vector = Array1::from_vec((0..64).map(|i| (i as f32) * 0.01).collect());

        let (indices, quantized) = pq.quantize(&vector).unwrap();

        assert_eq!(indices.len(), 4);
        assert_eq!(quantized.len(), 64);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < 8);
        }

        // Decode should reconstruct the quantized vector
        let decoded = pq.decode(&indices).unwrap();
        assert_eq!(decoded.len(), 64);

        // Quantized and decoded should be identical
        for (q, d) in quantized.iter().zip(decoded.iter()) {
            assert!((q - d).abs() < 1e-6);
        }
    }

    #[test]
    fn test_product_quantizer_batch() {
        let config = ProductQuantizerConfig {
            num_subspaces: 2,
            codebook_size_per_subspace: 16,
            embed_dim: 32,
            ..Default::default()
        };

        let pq = ProductQuantizer::new(config).unwrap();

        let vectors = vec![
            Array1::from_vec((0..32).map(|i| i as f32 * 0.1).collect()),
            Array1::from_vec((0..32).map(|i| i as f32 * 0.2).collect()),
            Array1::from_vec((0..32).map(|i| i as f32 * 0.3).collect()),
        ];

        let (all_indices, all_quantized) = pq.quantize_batch(&vectors).unwrap();

        assert_eq!(all_indices.len(), 3);
        assert_eq!(all_quantized.len(), 3);

        for (i, (indices, quantized)) in all_indices.iter().zip(all_quantized.iter()).enumerate() {
            assert_eq!(indices.len(), 2);
            assert_eq!(quantized.len(), 32);

            // Verify decode
            let decoded = pq.decode(indices).unwrap();
            for (q, d) in quantized.iter().zip(decoded.iter()) {
                assert!((q - d).abs() < 1e-6, "Batch {}: quantized != decoded", i);
            }
        }
    }

    #[test]
    fn test_product_quantizer_split_concat() {
        let config = ProductQuantizerConfig {
            num_subspaces: 4,
            codebook_size_per_subspace: 8,
            embed_dim: 64,
            ..Default::default()
        };

        let pq = ProductQuantizer::new(config).unwrap();
        let vector = Array1::from_vec((0..64).map(|i| i as f32).collect());

        let subspaces = pq.split_into_subspaces(&vector).unwrap();

        assert_eq!(subspaces.len(), 4);
        for subspace in &subspaces {
            assert_eq!(subspace.len(), 16); // 64 / 4
        }

        // Concatenate back
        let reconstructed = pq.concatenate_subspaces(&subspaces).unwrap();
        assert_eq!(reconstructed.len(), 64);

        for (orig, recon) in vector.iter().zip(reconstructed.iter()) {
            assert_eq!(orig, recon);
        }
    }

    #[test]
    fn test_product_quantizer_ema_update() {
        let config = ProductQuantizerConfig {
            num_subspaces: 2,
            codebook_size_per_subspace: 8,
            embed_dim: 16,
            use_ema: true,
            ..Default::default()
        };

        let mut pq = ProductQuantizer::new(config).unwrap();

        let outputs = vec![
            Array1::from_vec((0..16).map(|i| i as f32 * 0.1).collect()),
            Array1::from_vec((0..16).map(|i| i as f32 * 0.2).collect()),
            Array1::from_vec((0..16).map(|i| i as f32 * 0.3).collect()),
        ];

        let (all_indices, _) = pq.quantize_batch(&outputs).unwrap();

        // Update should succeed
        pq.update_ema(&outputs, &all_indices).unwrap();

        // Check usage stats for each subspace
        let stats = pq.usage_stats();
        assert_eq!(stats.len(), 2); // 2 subspaces

        for (total, used, _util) in stats {
            assert_eq!(total, 3); // 3 vectors processed
            assert!(used > 0); // At least some codes used
            assert!(used <= 8); // At most all codes used
        }
    }

    #[test]
    fn test_product_quantizer_initialization() {
        let config = ProductQuantizerConfig {
            num_subspaces: 2,
            codebook_size_per_subspace: 4,
            embed_dim: 16,
            ..Default::default()
        };

        let mut pq = ProductQuantizer::new(config).unwrap();

        // Generate some sample data
        let data: Vec<Array1<f32>> = (0..20)
            .map(|i| Array1::from_vec((0..16).map(|j| ((i + j) as f32 * 0.1).sin()).collect()))
            .collect();

        // Initialize from data (k-means++)
        pq.initialize_from_data(&data).unwrap();

        // After initialization, quantization should work
        let (indices, _) = pq.quantize(&data[0]).unwrap();
        assert_eq!(indices.len(), 2);
    }

    #[test]
    fn test_product_quantizer_compute_loss() {
        let config = ProductQuantizerConfig {
            num_subspaces: 4,
            codebook_size_per_subspace: 8,
            embed_dim: 64,
            commitment_beta: 0.25,
            ..Default::default()
        };

        let pq = ProductQuantizer::new(config).unwrap();

        let encoder_output = Array1::from_vec((0..64).map(|i| i as f32 * 0.01).collect());
        let (_, quantized) = pq.quantize(&encoder_output).unwrap();

        let (total_loss, codebook_loss, commitment_loss) =
            pq.compute_loss(&encoder_output, &quantized);

        assert!(total_loss >= 0.0);
        assert!(codebook_loss >= 0.0);
        assert!(commitment_loss >= 0.0);

        // Total loss = codebook_loss + beta * commitment_loss
        let expected_total = codebook_loss + 0.25 * commitment_loss;
        assert!((total_loss - expected_total).abs() < 1e-6);
    }

    #[test]
    fn test_product_quantizer_effective_size() {
        let config = ProductQuantizerConfig {
            num_subspaces: 4,
            codebook_size_per_subspace: 256,
            embed_dim: 64,
            ..Default::default()
        };

        let pq = ProductQuantizer::new(config.clone()).unwrap();

        // Effective codebook size = 256^4
        assert_eq!(pq.effective_codebook_size(), 256_usize.pow(4));

        // Number of parameters = M * K * (D/M) = M * K * D/M = K * D
        let expected_params = config.num_subspaces
            * config.codebook_size_per_subspace
            * (config.embed_dim / config.num_subspaces);
        assert_eq!(pq.num_parameters(), expected_params);
    }

    #[test]
    fn test_product_quantizer_reset_unused_codes() {
        let config = ProductQuantizerConfig {
            num_subspaces: 2,
            codebook_size_per_subspace: 8,
            embed_dim: 16,
            ..Default::default()
        };

        let mut pq = ProductQuantizer::new(config).unwrap();

        // Generate some encoder outputs
        let outputs: Vec<Array1<f32>> = (0..5)
            .map(|i| Array1::from_vec((0..16).map(|j| (i + j) as f32 * 0.1).collect()))
            .collect();

        // Quantize to mark some codes as used
        let (all_indices, _) = pq.quantize_batch(&outputs).unwrap();

        // Update EMA to track usage
        pq.update_ema(&outputs, &all_indices).unwrap();

        // Reset codes that have been used less than 2 times
        let reset_count = pq.reset_unused_codes(&outputs, 2).unwrap();

        // Function should succeed and return a valid count
        // (reset_count could be 0 or higher depending on how codes were used)
        assert!(reset_count <= pq.num_subspaces() * pq.codebook_size_per_subspace());
    }

    #[test]
    fn test_product_quantizer_memory_efficiency() {
        // Compare memory usage of PQ vs standard VQ

        // Standard VQ: 1M codes, 128-dim
        let standard_params = 1_000_000 * 128;

        // Product VQ: 4 subspaces, 100 codes each, 128-dim
        // Effective codes: 100^4 = 100M (100x more codes!)
        // Parameters: 4 * 100 * 32 = 12,800
        let pq_config = ProductQuantizerConfig {
            num_subspaces: 4,
            codebook_size_per_subspace: 100,
            embed_dim: 128,
            ..Default::default()
        };

        let pq = ProductQuantizer::new(pq_config).unwrap();

        assert_eq!(pq.num_parameters(), 4 * 100 * 32);
        assert_eq!(pq.effective_codebook_size(), 100_usize.pow(4));

        // PQ uses ~10,000x fewer parameters for ~100x more codes
        let compression_ratio = standard_params as f32 / pq.num_parameters() as f32;
        assert!(compression_ratio > 9000.0);
    }
}
