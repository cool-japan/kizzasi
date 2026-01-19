//! Neural network building blocks: normalization and activation functions
//!
//! Provides layer normalization variants and gating mechanisms for SSM architectures.

use scirs2_core::ndarray::Array1;

// ============================================================================
// Layer Normalization
// ============================================================================

/// Type of normalization to apply
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormType {
    /// Standard Layer Normalization (Ba et al., 2016)
    LayerNorm,
    /// RMS Layer Normalization (Zhang & Sennrich, 2019)
    #[default]
    RMSNorm, // RMSNorm is faster and commonly used in modern SSMs
    /// No normalization
    None,
}

/// Layer normalization with learnable parameters
#[derive(Debug, Clone)]
pub struct LayerNorm {
    gamma: Array1<f32>, // scale
    beta: Array1<f32>,  // shift
    eps: f32,
    norm_type: NormType,
}

impl LayerNorm {
    /// Create a new LayerNorm
    pub fn new(dim: usize, norm_type: NormType) -> Self {
        Self {
            gamma: Array1::ones(dim),
            beta: Array1::zeros(dim),
            eps: 1e-5,
            norm_type,
        }
    }

    /// Create with custom epsilon
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Set gamma (scale) parameters
    pub fn set_gamma(&mut self, gamma: Array1<f32>) {
        self.gamma = gamma;
    }

    /// Set beta (shift) parameters
    pub fn set_beta(&mut self, beta: Array1<f32>) {
        self.beta = beta;
    }

    /// Apply normalization to input
    pub fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        match self.norm_type {
            NormType::LayerNorm => self.layer_norm(x),
            NormType::RMSNorm => self.rms_norm(x),
            NormType::None => x.clone(),
        }
    }

    /// Standard layer normalization: (x - mean) / std * gamma + beta
    fn layer_norm(&self, x: &Array1<f32>) -> Array1<f32> {
        let n = x.len() as f32;
        let mean = x.sum() / n;
        let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
        let std = (var + self.eps).sqrt();

        let mut result = Array1::zeros(x.len());
        for i in 0..x.len() {
            result[i] = ((x[i] - mean) / std) * self.gamma[i] + self.beta[i];
        }
        result
    }

    /// RMS layer normalization: x / rms(x) * gamma
    fn rms_norm(&self, x: &Array1<f32>) -> Array1<f32> {
        let n = x.len() as f32;
        let rms = (x.iter().map(|&v| v * v).sum::<f32>() / n + self.eps).sqrt();

        let mut result = Array1::zeros(x.len());
        for i in 0..x.len() {
            result[i] = (x[i] / rms) * self.gamma[i];
        }
        result
    }

    /// Get norm type
    pub fn norm_type(&self) -> NormType {
        self.norm_type
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.gamma.len()
    }
}

/// Standalone layer normalization function
pub fn layer_norm(x: &Array1<f32>, eps: f32) -> Array1<f32> {
    let n = x.len() as f32;
    let mean = x.sum() / n;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();

    let mut result = Array1::zeros(x.len());
    for i in 0..x.len() {
        result[i] = (x[i] - mean) / std;
    }
    result
}

/// Standalone RMS normalization function
pub fn rms_norm(x: &Array1<f32>, eps: f32) -> Array1<f32> {
    let n = x.len() as f32;
    let rms = (x.iter().map(|&v| v * v).sum::<f32>() / n + eps).sqrt();

    let mut result = Array1::zeros(x.len());
    for i in 0..x.len() {
        result[i] = x[i] / rms;
    }
    result
}

// ============================================================================
// Activation Functions (Gating Mechanisms)
// ============================================================================

/// Type of activation function
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActivationType {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Gaussian Error Linear Unit: x * Phi(x)
    GELU,
    /// Sigmoid Linear Unit (Swish): x * sigmoid(x)
    #[default]
    SiLU, // SiLU is commonly used in Mamba
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// No activation (identity)
    None,
}

/// Gated activation with configurable type
#[derive(Debug, Clone)]
pub struct Activation {
    act_type: ActivationType,
}

impl Activation {
    /// Create a new activation
    pub fn new(act_type: ActivationType) -> Self {
        Self { act_type }
    }

    /// Apply activation to input
    pub fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        match self.act_type {
            ActivationType::ReLU => relu(x),
            ActivationType::GELU => gelu(x),
            ActivationType::SiLU => silu(x),
            ActivationType::Sigmoid => sigmoid(x),
            ActivationType::Tanh => tanh(x),
            ActivationType::None => x.clone(),
        }
    }

    /// Get activation type
    pub fn act_type(&self) -> ActivationType {
        self.act_type
    }
}

/// ReLU activation: max(0, x)
pub fn relu(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v.max(0.0))
}

/// Leaky ReLU: max(alpha * x, x)
pub fn leaky_relu(x: &Array1<f32>, alpha: f32) -> Array1<f32> {
    x.mapv(|v| if v >= 0.0 { v } else { alpha * v })
}

/// Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

/// Tanh activation
pub fn tanh(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v.tanh())
}

/// SiLU (Swish) activation: x * sigmoid(x)
///
/// Commonly used in Mamba and modern SSM architectures
pub fn silu(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v / (1.0 + (-v).exp()))
}

/// GELU activation: x * Phi(x) where Phi is the CDF of standard normal
///
/// Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: &Array1<f32>) -> Array1<f32> {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/pi)
    const COEF: f32 = 0.044715;

    x.mapv(|v| {
        let inner = SQRT_2_OVER_PI * (v + COEF * v.powi(3));
        0.5 * v * (1.0 + inner.tanh())
    })
}

/// Fast GELU approximation using sigmoid
pub fn gelu_fast(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v / (1.0 + (-1.702 * v).exp()))
}

/// Softmax: exp(x_i) / sum(exp(x))
pub fn softmax(x: &Array1<f32>) -> Array1<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_x: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exp_x.iter().sum();
    Array1::from_vec(exp_x.iter().map(|&v| v / sum).collect())
}

/// Log softmax: log(softmax(x)) - more numerically stable
pub fn log_softmax(x: &Array1<f32>) -> Array1<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Array1<f32> = x.mapv(|v| v - max_val);
    let log_sum_exp = shifted.mapv(|v| v.exp()).sum().ln();
    shifted.mapv(|v| v - log_sum_exp)
}

// ============================================================================
// Gated Linear Unit
// ============================================================================

/// Gated Linear Unit: splits input in half and applies gate
///
/// GLU(x, W, V) = (xW + b) * sigmoid(xV + c)
#[derive(Debug, Clone)]
pub struct GatedLinearUnit {
    /// Activation for the gate
    gate_activation: ActivationType,
}

impl GatedLinearUnit {
    /// Create a new GLU with default sigmoid gate
    pub fn new() -> Self {
        Self {
            gate_activation: ActivationType::Sigmoid,
        }
    }

    /// Create with SiLU gate (SwiGLU, commonly used in modern architectures)
    pub fn swiglu() -> Self {
        Self {
            gate_activation: ActivationType::SiLU,
        }
    }

    /// Create with GELU gate (GeGLU)
    pub fn geglu() -> Self {
        Self {
            gate_activation: ActivationType::GELU,
        }
    }

    /// Apply GLU to input (input should have even dimension)
    ///
    /// Splits input into [x, gate] and returns x * activation(gate)
    pub fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        let n = x.len();
        if n < 2 {
            return x.clone();
        }

        let half = n / 2;
        let x_part: Array1<f32> = Array1::from_vec(x.iter().take(half).cloned().collect());
        let gate_part: Array1<f32> =
            Array1::from_vec(x.iter().skip(half).take(half).cloned().collect());

        let gate = match self.gate_activation {
            ActivationType::Sigmoid => sigmoid(&gate_part),
            ActivationType::SiLU => silu(&gate_part),
            ActivationType::GELU => gelu(&gate_part),
            _ => sigmoid(&gate_part),
        };

        &x_part * &gate
    }
}

impl Default for GatedLinearUnit {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let norm = LayerNorm::new(4, NormType::LayerNorm);
        let y = norm.forward(&x);

        // After normalization, mean should be ~0, std should be ~1
        let mean: f32 = y.sum() / y.len() as f32;
        assert!(mean.abs() < 0.01);
    }

    #[test]
    fn test_rms_norm() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let norm = LayerNorm::new(4, NormType::RMSNorm);
        let y = norm.forward(&x);

        // RMS of output should be ~1 (scaled by gamma=1)
        let rms = (y.iter().map(|v| v * v).sum::<f32>() / y.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_relu() {
        let x = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let y = relu(&x);
        assert_eq!(y[0], 0.0);
        assert_eq!(y[1], 0.0);
        assert_eq!(y[2], 0.0);
        assert_eq!(y[3], 1.0);
        assert_eq!(y[4], 2.0);
    }

    #[test]
    fn test_sigmoid() {
        let x = Array1::from_vec(vec![-10.0, 0.0, 10.0]);
        let y = sigmoid(&x);
        assert!(y[0] < 0.01); // sigmoid(-10) ≈ 0
        assert!((y[1] - 0.5).abs() < 0.01); // sigmoid(0) = 0.5
        assert!(y[2] > 0.99); // sigmoid(10) ≈ 1
    }

    #[test]
    fn test_silu() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let y = silu(&x);
        assert!((y[0] - 0.0).abs() < 0.01); // silu(0) = 0 * 0.5 = 0
        assert!((y[1] - 0.731).abs() < 0.01); // silu(1) ≈ 0.731
    }

    #[test]
    fn test_gelu() {
        let x = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        let y = gelu(&x);
        assert!((y[1] - 0.0).abs() < 0.01); // gelu(0) = 0
        assert!(y[2] > 0.5); // gelu(1) > 0.5
        assert!(y[0] < 0.0); // gelu(-1) < 0
    }

    #[test]
    fn test_softmax() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = softmax(&x);

        // Sum should be 1
        assert!((y.sum() - 1.0).abs() < 0.01);
        // Values should be ordered
        assert!(y[2] > y[1] && y[1] > y[0]);
    }

    #[test]
    fn test_glu() {
        let x = Array1::from_vec(vec![1.0, 2.0, 0.0, 0.0]); // [data, gate]
        let glu = GatedLinearUnit::new();
        let y = glu.forward(&x);

        assert_eq!(y.len(), 2);
        // gate = sigmoid([0, 0]) = [0.5, 0.5]
        // result = [1, 2] * [0.5, 0.5] = [0.5, 1.0]
        assert!((y[0] - 0.5).abs() < 0.01);
        assert!((y[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_swiglu() {
        let x = Array1::from_vec(vec![1.0, 2.0, 1.0, 1.0]);
        let glu = GatedLinearUnit::swiglu();
        let y = glu.forward(&x);

        assert_eq!(y.len(), 2);
        // SiLU gate applied
        assert!(y[0] > 0.0);
        assert!(y[1] > 0.0);
    }
}
