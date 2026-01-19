//! # RWKV-7 Architecture
//!
//! Receptance Weighted Key Value (RWKV-7) - Latest version of the RWKV architecture.
//!
//! ## Features
//!
//! - **Linear Time Complexity**: O(1) per-step inference like RNNs
//! - **Time-Mixing**: Attention-like mechanism with learnable interpolation
//! - **Channel-Mixing**: Feed-forward with time-dependent gating
//! - **LayerNorm**: Pre-normalization for stable training
//! - **WKV State**: Weighted key-value state for efficient recurrence
//!
//! ## Architecture
//!
//! RWKV-7 consists of stacked layers, each containing:
//! 1. Time-mixing block (attention replacement)
//! 2. Channel-mixing block (FFN replacement)
//! 3. LayerNorm before each block
//!
//! ## References
//!
//! - RWKV-7 paper: "RWKV: Reinventing RNNs for the Transformer Era"
//! - https://github.com/BlinkDL/RWKV-LM

use crate::{CoreError, CoreResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::quick::random_f32;
use serde::{Deserialize, Serialize};

/// RWKV-7 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RWKV7Config {
    /// Input dimension
    pub d_input: usize,
    /// Model dimension (embedding size)
    pub d_model: usize,
    /// Number of layers
    pub n_layers: usize,
    /// FFN expansion factor (typically 3.5 or 4)
    pub ffn_factor: f32,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
    /// Time-decay initial value
    pub time_decay_init: f32,
    /// Time-first initial value
    pub time_first_init: f32,
}

impl Default for RWKV7Config {
    fn default() -> Self {
        Self {
            d_input: 128,
            d_model: 256,
            n_layers: 6,
            ffn_factor: 3.5,
            layer_norm_eps: 1e-5,
            time_decay_init: -5.0,
            time_first_init: 0.0,
        }
    }
}

impl RWKV7Config {
    /// Create a new RWKV-7 configuration
    pub fn new(d_input: usize, d_model: usize, n_layers: usize) -> Self {
        Self {
            d_input,
            d_model,
            n_layers,
            ..Default::default()
        }
    }

    /// Set FFN expansion factor
    pub fn with_ffn_factor(mut self, factor: f32) -> Self {
        self.ffn_factor = factor;
        self
    }

    /// Set layer norm epsilon
    pub fn with_layer_norm_eps(mut self, eps: f32) -> Self {
        self.layer_norm_eps = eps;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> CoreResult<()> {
        if self.d_input == 0 || self.d_model == 0 || self.n_layers == 0 {
            return Err(CoreError::InvalidConfig(
                "Dimensions and layers must be positive".to_string(),
            ));
        }
        if self.ffn_factor <= 0.0 {
            return Err(CoreError::InvalidConfig(
                "FFN factor must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Time-mixing block (attention replacement)
///
/// Implements the core RWKV mechanism: receptance-weighted key-value
pub struct TimeMixing {
    d_model: usize,
    // Learnable time-mixing parameters
    time_mix_k: Array1<f32>,
    time_mix_v: Array1<f32>,
    time_mix_r: Array1<f32>,
    // Time decay (learnable)
    time_decay: Array1<f32>,
    time_first: Array1<f32>,
    // Projection weights
    key_w: Array2<f32>,
    value_w: Array2<f32>,
    receptance_w: Array2<f32>,
    output_w: Array2<f32>,
    // WKV state
    wkv_state: Array2<f32>, // (d_model, 2) - stores (numerator, denominator)
    prev_x: Array1<f32>,
}

impl TimeMixing {
    /// Create a new time-mixing block
    pub fn new(config: &RWKV7Config) -> Self {
        let d_model = config.d_model;

        // Initialize time-mixing parameters (learnable interpolation)
        let time_mix_k = Array1::from_elem(d_model, 0.5);
        let time_mix_v = Array1::from_elem(d_model, 0.5);
        let time_mix_r = Array1::from_elem(d_model, 0.5);

        // Initialize time decay (log-space for stability)
        let time_decay = Array1::from_elem(d_model, config.time_decay_init);
        let time_first = Array1::from_elem(d_model, config.time_first_init);

        // Initialize projection weights (Xavier initialization)
        let scale = (2.0 / d_model as f32).sqrt();
        let key_w =
            Array2::from_shape_fn((d_model, d_model), |_| (random_f32() - 0.5) * 2.0 * scale);
        let value_w =
            Array2::from_shape_fn((d_model, d_model), |_| (random_f32() - 0.5) * 2.0 * scale);
        let receptance_w =
            Array2::from_shape_fn((d_model, d_model), |_| (random_f32() - 0.5) * 2.0 * scale);
        let output_w =
            Array2::from_shape_fn((d_model, d_model), |_| (random_f32() - 0.5) * 2.0 * scale);

        let wkv_state = Array2::zeros((d_model, 2));
        let prev_x = Array1::zeros(d_model);

        Self {
            d_model,
            time_mix_k,
            time_mix_v,
            time_mix_r,
            time_decay,
            time_first,
            key_w,
            value_w,
            receptance_w,
            output_w,
            wkv_state,
            prev_x,
        }
    }

    /// Forward pass with WKV computation
    pub fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        if x.len() != self.d_model {
            return Err(CoreError::DimensionMismatch {
                expected: self.d_model,
                got: x.len(),
            });
        }

        // Time-mixing: interpolate between current and previous input
        let mut k_input = Array1::zeros(self.d_model);
        let mut v_input = Array1::zeros(self.d_model);
        let mut r_input = Array1::zeros(self.d_model);

        for i in 0..self.d_model {
            k_input[i] = self.time_mix_k[i] * x[i] + (1.0 - self.time_mix_k[i]) * self.prev_x[i];
            v_input[i] = self.time_mix_v[i] * x[i] + (1.0 - self.time_mix_v[i]) * self.prev_x[i];
            r_input[i] = self.time_mix_r[i] * x[i] + (1.0 - self.time_mix_r[i]) * self.prev_x[i];
        }

        // Compute K, V, R projections
        let k = self.key_w.dot(&k_input);
        let v = self.value_w.dot(&v_input);
        let r = self.receptance_w.dot(&r_input);

        // WKV: Weighted Key-Value mechanism
        let wkv = self.compute_wkv(&k, &v)?;

        // Apply receptance and output projection
        let mut rwkv = Array1::zeros(self.d_model);
        for i in 0..self.d_model {
            rwkv[i] = self.sigmoid(r[i]) * wkv[i];
        }

        let output = self.output_w.dot(&rwkv);

        // Update state
        self.prev_x = x.clone();

        Ok(output)
    }

    /// Compute WKV (Weighted Key-Value) state update
    fn compute_wkv(&mut self, k: &Array1<f32>, v: &Array1<f32>) -> CoreResult<Array1<f32>> {
        let mut wkv = Array1::zeros(self.d_model);

        for i in 0..self.d_model {
            // Get previous state
            let prev_num = self.wkv_state[[i, 0]];
            let prev_den = self.wkv_state[[i, 1]];

            // Compute decay (in log-space for numerical stability)
            let w = (-self.time_decay[i].exp()).exp(); // e^(-e^(time_decay))
            let u = self.time_first[i].exp();

            // Update state with time decay
            // numerator = w * prev_num + u * k[i] * v[i]
            // denominator = w * prev_den + u * k[i]
            let new_num = w * prev_num + u * k[i] * v[i];
            let new_den = w * prev_den + u * k[i];

            // Compute output (numerator / denominator)
            wkv[i] = if new_den.abs() > 1e-8 {
                new_num / new_den
            } else {
                0.0
            };

            // Store new state
            self.wkv_state[[i, 0]] = new_num;
            self.wkv_state[[i, 1]] = new_den;
        }

        Ok(wkv)
    }

    /// Sigmoid activation
    fn sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.wkv_state.fill(0.0);
        self.prev_x.fill(0.0);
    }
}

/// Channel-mixing block (FFN replacement)
///
/// Implements position-wise feed-forward with time-dependent gating
pub struct ChannelMixing {
    d_model: usize,
    #[allow(dead_code)] // Used in tests
    d_ffn: usize,
    // Time-mixing parameter
    time_mix_k: Array1<f32>,
    time_mix_r: Array1<f32>,
    // FFN weights
    key_w: Array2<f32>,
    value_w: Array2<f32>,
    receptance_w: Array2<f32>,
    // Previous input
    prev_x: Array1<f32>,
}

impl ChannelMixing {
    /// Create a new channel-mixing block
    pub fn new(config: &RWKV7Config) -> Self {
        let d_model = config.d_model;
        let d_ffn = (d_model as f32 * config.ffn_factor) as usize;

        let time_mix_k = Array1::from_elem(d_model, 0.5);
        let time_mix_r = Array1::from_elem(d_model, 0.5);

        let scale = (2.0 / d_model as f32).sqrt();
        let key_w = Array2::from_shape_fn((d_ffn, d_model), |_| (random_f32() - 0.5) * 2.0 * scale);
        let value_w =
            Array2::from_shape_fn((d_model, d_ffn), |_| (random_f32() - 0.5) * 2.0 * scale);
        let receptance_w =
            Array2::from_shape_fn((d_model, d_model), |_| (random_f32() - 0.5) * 2.0 * scale);

        let prev_x = Array1::zeros(d_model);

        Self {
            d_model,
            d_ffn,
            time_mix_k,
            time_mix_r,
            key_w,
            value_w,
            receptance_w,
            prev_x,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        if x.len() != self.d_model {
            return Err(CoreError::DimensionMismatch {
                expected: self.d_model,
                got: x.len(),
            });
        }

        // Time-mixing
        let mut k_input = Array1::zeros(self.d_model);
        let mut r_input = Array1::zeros(self.d_model);

        for i in 0..self.d_model {
            k_input[i] = self.time_mix_k[i] * x[i] + (1.0 - self.time_mix_k[i]) * self.prev_x[i];
            r_input[i] = self.time_mix_r[i] * x[i] + (1.0 - self.time_mix_r[i]) * self.prev_x[i];
        }

        // FFN with squared ReLU activation
        let k = self.key_w.dot(&k_input);
        let kv = self.apply_squared_relu(&k);
        let v = self.value_w.dot(&kv);

        // Receptance gating
        let r = self.receptance_w.dot(&r_input);
        let mut output = Array1::zeros(self.d_model);
        for i in 0..self.d_model {
            output[i] = self.sigmoid(r[i]) * v[i];
        }

        // Update state
        self.prev_x = x.clone();

        Ok(output)
    }

    /// Squared ReLU activation (x^2 for x > 0, else 0)
    fn apply_squared_relu(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| if v > 0.0 { v * v } else { 0.0 })
    }

    /// Sigmoid activation
    fn sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.prev_x.fill(0.0);
    }
}

/// Single RWKV-7 layer
pub struct RWKV7Layer {
    config: RWKV7Config,
    time_mixing: TimeMixing,
    channel_mixing: ChannelMixing,
    ln1_weight: Array1<f32>,
    ln1_bias: Array1<f32>,
    ln2_weight: Array1<f32>,
    ln2_bias: Array1<f32>,
}

impl RWKV7Layer {
    /// Create a new RWKV-7 layer
    pub fn new(config: RWKV7Config) -> CoreResult<Self> {
        config.validate()?;

        let time_mixing = TimeMixing::new(&config);
        let channel_mixing = ChannelMixing::new(&config);

        let ln1_weight = Array1::ones(config.d_model);
        let ln1_bias = Array1::zeros(config.d_model);
        let ln2_weight = Array1::ones(config.d_model);
        let ln2_bias = Array1::zeros(config.d_model);

        Ok(Self {
            config,
            time_mixing,
            channel_mixing,
            ln1_weight,
            ln1_bias,
            ln2_weight,
            ln2_bias,
        })
    }

    /// Forward pass through the layer
    pub fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // Time-mixing block with residual
        let x_norm1 = self.layer_norm(x, &self.ln1_weight, &self.ln1_bias)?;
        let tm_out = self.time_mixing.forward(&x_norm1)?;
        let x = &(x + &tm_out);

        // Channel-mixing block with residual
        let x_norm2 = self.layer_norm(x, &self.ln2_weight, &self.ln2_bias)?;
        let cm_out = self.channel_mixing.forward(&x_norm2)?;
        let output = x + &cm_out;

        Ok(output)
    }

    /// Layer normalization
    fn layer_norm(
        &self,
        x: &Array1<f32>,
        weight: &Array1<f32>,
        bias: &Array1<f32>,
    ) -> CoreResult<Array1<f32>> {
        let mean = x.mean().unwrap_or(0.0);
        let var = x.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(0.0);
        let std = (var + self.config.layer_norm_eps).sqrt();

        let mut normalized = Array1::zeros(x.len());
        for i in 0..x.len() {
            normalized[i] = ((x[i] - mean) / std) * weight[i] + bias[i];
        }

        Ok(normalized)
    }

    /// Reset layer state
    pub fn reset(&mut self) {
        self.time_mixing.reset();
        self.channel_mixing.reset();
    }
}

/// RWKV-7 model (stacked layers)
pub struct RWKV7Model {
    config: RWKV7Config,
    embedding: Array2<f32>,
    layers: Vec<RWKV7Layer>,
    ln_out_weight: Array1<f32>,
    ln_out_bias: Array1<f32>,
}

impl RWKV7Model {
    /// Create a new RWKV-7 model
    pub fn new(config: RWKV7Config) -> CoreResult<Self> {
        config.validate()?;

        // Initialize embedding
        let scale = (1.0 / config.d_input as f32).sqrt();
        let embedding = Array2::from_shape_fn((config.d_model, config.d_input), |_| {
            (random_f32() - 0.5) * 2.0 * scale
        });

        // Create layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            layers.push(RWKV7Layer::new(config.clone())?);
        }

        let ln_out_weight = Array1::ones(config.d_model);
        let ln_out_bias = Array1::zeros(config.d_model);

        Ok(Self {
            config,
            embedding,
            layers,
            ln_out_weight,
            ln_out_bias,
        })
    }

    /// Forward pass through the model
    pub fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        if x.len() != self.config.d_input {
            return Err(CoreError::DimensionMismatch {
                expected: self.config.d_input,
                got: x.len(),
            });
        }

        // Embedding
        let mut h = self.embedding.dot(x);

        // Process through layers
        for layer in &mut self.layers {
            h = layer.forward(&h)?;
        }

        // Final layer norm
        let output = self.layer_norm(&h)?;

        Ok(output)
    }

    /// Layer normalization
    fn layer_norm(&self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        let mean = x.mean().unwrap_or(0.0);
        let var = x.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(0.0);
        let std = (var + self.config.layer_norm_eps).sqrt();

        let mut normalized = Array1::zeros(x.len());
        for i in 0..x.len() {
            normalized[i] = ((x[i] - mean) / std) * self.ln_out_weight[i] + self.ln_out_bias[i];
        }

        Ok(normalized)
    }

    /// Reset all layer states
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Get model configuration
    pub fn config(&self) -> &RWKV7Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rwkv7_config() {
        let config = RWKV7Config::new(64, 128, 4);
        assert_eq!(config.d_input, 64);
        assert_eq!(config.d_model, 128);
        assert_eq!(config.n_layers, 4);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_rwkv7_config_validation() {
        let config = RWKV7Config {
            d_model: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_time_mixing_creation() {
        let config = RWKV7Config::new(64, 128, 2);
        let tm = TimeMixing::new(&config);
        assert_eq!(tm.d_model, 128);
        assert_eq!(tm.time_mix_k.len(), 128);
    }

    #[test]
    fn test_time_mixing_forward() {
        let config = RWKV7Config::new(64, 128, 2);
        let mut tm = TimeMixing::new(&config);
        let x = Array1::from_elem(128, 0.5);
        let result = tm.forward(&x);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 128);
    }

    #[test]
    fn test_channel_mixing_creation() {
        let config = RWKV7Config::new(64, 128, 2);
        let cm = ChannelMixing::new(&config);
        assert_eq!(cm.d_model, 128);
        assert_eq!(cm.d_ffn, (128.0 * 3.5) as usize);
    }

    #[test]
    fn test_channel_mixing_forward() {
        let config = RWKV7Config::new(64, 128, 2);
        let mut cm = ChannelMixing::new(&config);
        let x = Array1::from_elem(128, 0.3);
        let result = cm.forward(&x);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 128);
    }

    #[test]
    fn test_rwkv7_layer_creation() {
        let config = RWKV7Config::new(64, 128, 2);
        let layer = RWKV7Layer::new(config);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_rwkv7_layer_forward() {
        let config = RWKV7Config::new(64, 128, 2);
        let mut layer = RWKV7Layer::new(config).unwrap();
        let x = Array1::from_elem(128, 0.1);
        let result = layer.forward(&x);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 128);
        assert!(output.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_rwkv7_model_creation() {
        let config = RWKV7Config::new(64, 128, 4);
        let model = RWKV7Model::new(config);
        assert!(model.is_ok());
        let m = model.unwrap();
        assert_eq!(m.layers.len(), 4);
    }

    #[test]
    fn test_rwkv7_model_forward() {
        let config = RWKV7Config::new(64, 128, 3);
        let mut model = RWKV7Model::new(config).unwrap();
        let x = Array1::from_elem(64, 0.2);
        let result = model.forward(&x);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 128);
        assert!(output.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_rwkv7_reset() {
        let config = RWKV7Config::new(64, 128, 2);
        let mut model = RWKV7Model::new(config).unwrap();

        // Run forward pass
        let x = Array1::from_elem(64, 0.5);
        let _ = model.forward(&x).unwrap();

        // Reset
        model.reset();

        // States should be cleared
        for layer in &model.layers {
            assert!(layer.time_mixing.wkv_state.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn test_wkv_mechanism() {
        let config = RWKV7Config::new(64, 128, 2);
        let mut tm = TimeMixing::new(&config);

        // Process sequence
        let x1 = Array1::from_elem(128, 0.1);
        let x2 = Array1::from_elem(128, 0.5);
        let x3 = Array1::from_elem(128, 0.9);

        let _ = tm.forward(&x1).unwrap();
        let _ = tm.forward(&x2).unwrap();
        let out3 = tm.forward(&x3).unwrap();

        // State should be non-zero and finite
        assert!(tm.wkv_state.iter().any(|&v| v != 0.0));
        assert!(out3.iter().all(|&v| v.is_finite()));
    }
}
