//! Weight Quantization for Efficient Inference
//!
//! Provides INT8 quantization for model weights to reduce memory usage
//! and improve inference speed on integer-optimized hardware.
//!
//! # Features
//!
//! - **INT8 quantization**: Symmetric and asymmetric quantization
//! - **Per-tensor quantization**: Single scale/zero-point for entire tensor
//! - **Per-channel quantization**: Independent scale/zero-point per output channel
//! - **Calibration**: Automatic scale/zero-point computation from data
//! - **Mixed precision**: Selective quantization of layers
//!
//! # Theory
//!
//! Quantization maps floating-point values to integers:
//! ```text
//! q = round(x / scale) + zero_point
//! x_approx = (q - zero_point) * scale
//! ```
//!
//! For symmetric quantization (zero_point = 0):
//! ```text
//! scale = max(|x|) / 127
//! q = clamp(round(x / scale), -128, 127)
//! ```

use crate::error::{ModelError, ModelResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Quantization method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMethod {
    /// Symmetric quantization (zero_point = 0)
    Symmetric,
    /// Asymmetric quantization (arbitrary zero_point)
    Asymmetric,
}

/// Quantization granularity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationGranularity {
    /// Single scale/zero_point for entire tensor
    PerTensor,
    /// Independent scale/zero_point per output channel (dim 0)
    PerChannel,
}

/// Quantization parameters
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// Scale factor(s)
    pub scale: Vec<f32>,
    /// Zero point(s)
    pub zero_point: Vec<i8>,
    /// Quantization method
    pub method: QuantizationMethod,
    /// Granularity
    pub granularity: QuantizationGranularity,
}

impl QuantizationParams {
    /// Create symmetric per-tensor quantization params
    pub fn symmetric_per_tensor(scale: f32) -> Self {
        Self {
            scale: vec![scale],
            zero_point: vec![0],
            method: QuantizationMethod::Symmetric,
            granularity: QuantizationGranularity::PerTensor,
        }
    }

    /// Create asymmetric per-tensor quantization params
    pub fn asymmetric_per_tensor(scale: f32, zero_point: i8) -> Self {
        Self {
            scale: vec![scale],
            zero_point: vec![zero_point],
            method: QuantizationMethod::Asymmetric,
            granularity: QuantizationGranularity::PerTensor,
        }
    }

    /// Create symmetric per-channel quantization params
    pub fn symmetric_per_channel(scales: Vec<f32>) -> Self {
        let n = scales.len();
        Self {
            scale: scales,
            zero_point: vec![0; n],
            method: QuantizationMethod::Symmetric,
            granularity: QuantizationGranularity::PerChannel,
        }
    }

    /// Validate parameters
    pub fn validate(&self) -> ModelResult<()> {
        if self.scale.is_empty() {
            return Err(ModelError::invalid_config("scale cannot be empty"));
        }
        if self.scale.len() != self.zero_point.len() {
            return Err(ModelError::invalid_config(
                "scale and zero_point must have same length",
            ));
        }
        for &s in &self.scale {
            if s <= 0.0 || !s.is_finite() {
                return Err(ModelError::invalid_config(format!("invalid scale: {}", s)));
            }
        }
        Ok(())
    }
}

/// Quantized weight tensor
#[derive(Debug, Clone)]
pub struct QuantizedWeight {
    /// Quantized data (INT8)
    pub data: Vec<i8>,
    /// Original shape
    pub shape: Vec<usize>,
    /// Quantization parameters
    pub params: QuantizationParams,
}

impl QuantizedWeight {
    /// Create a new quantized weight
    pub fn new(data: Vec<i8>, shape: Vec<usize>, params: QuantizationParams) -> ModelResult<Self> {
        params.validate()?;

        let total_size: usize = shape.iter().product();
        if data.len() != total_size {
            return Err(ModelError::invalid_config(format!(
                "data length {} does not match shape {:?}",
                data.len(),
                shape
            )));
        }

        Ok(Self {
            data,
            shape,
            params,
        })
    }

    /// Dequantize to f32 array (1D)
    pub fn dequantize_1d(&self) -> ModelResult<Array1<f32>> {
        if self.shape.len() != 1 {
            return Err(ModelError::invalid_config(format!(
                "expected 1D shape, got {:?}",
                self.shape
            )));
        }

        let n = self.shape[0];
        let mut result = Array1::zeros(n);

        match self.params.granularity {
            QuantizationGranularity::PerTensor => {
                let scale = self.params.scale[0];
                let zero_point = self.params.zero_point[0];

                for i in 0..n {
                    result[i] = (self.data[i] as i32 - zero_point as i32) as f32 * scale;
                }
            }
            QuantizationGranularity::PerChannel => {
                return Err(ModelError::invalid_config(
                    "per-channel quantization not supported for 1D tensors",
                ));
            }
        }

        Ok(result)
    }

    /// Dequantize to f32 array (2D)
    pub fn dequantize_2d(&self) -> ModelResult<Array2<f32>> {
        if self.shape.len() != 2 {
            return Err(ModelError::invalid_config(format!(
                "expected 2D shape, got {:?}",
                self.shape
            )));
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut result = Array2::zeros((rows, cols));

        match self.params.granularity {
            QuantizationGranularity::PerTensor => {
                let scale = self.params.scale[0];
                let zero_point = self.params.zero_point[0];

                for i in 0..rows {
                    for j in 0..cols {
                        let idx = i * cols + j;
                        result[[i, j]] = (self.data[idx] as i32 - zero_point as i32) as f32 * scale;
                    }
                }
            }
            QuantizationGranularity::PerChannel => {
                // Per-channel quantization: independent scale/zero_point per row (output channel)
                if self.params.scale.len() != rows {
                    return Err(ModelError::invalid_config(format!(
                        "expected {} scales for per-channel, got {}",
                        rows,
                        self.params.scale.len()
                    )));
                }

                for i in 0..rows {
                    let scale = self.params.scale[i];
                    let zero_point = self.params.zero_point[i];

                    for j in 0..cols {
                        let idx = i * cols + j;
                        result[[i, j]] = (self.data[idx] as i32 - zero_point as i32) as f32 * scale;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Get memory size in bytes
    pub fn memory_size(&self) -> usize {
        self.data.len() // INT8: 1 byte per element
    }
}

/// Quantize f32 array to INT8 using symmetric quantization
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// Quantized weight with symmetric per-tensor quantization
pub fn quantize_symmetric_1d(array: &Array1<f32>) -> ModelResult<QuantizedWeight> {
    let max_val = array.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

    if max_val == 0.0 {
        // All zeros - use scale of 1.0 to avoid division by zero
        let data = vec![0i8; array.len()];
        let params = QuantizationParams::symmetric_per_tensor(1.0);
        return QuantizedWeight::new(data, vec![array.len()], params);
    }

    let scale = max_val / 127.0;
    let mut data = Vec::with_capacity(array.len());

    for &x in array.iter() {
        let q = (x / scale).round() as i32;
        let q_clamped = q.clamp(-128, 127) as i8;
        data.push(q_clamped);
    }

    let params = QuantizationParams::symmetric_per_tensor(scale);
    QuantizedWeight::new(data, vec![array.len()], params)
}

/// Quantize f32 2D array to INT8 using symmetric per-tensor quantization
pub fn quantize_symmetric_2d(array: &Array2<f32>) -> ModelResult<QuantizedWeight> {
    let max_val = array.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

    if max_val == 0.0 {
        let (rows, cols) = array.dim();
        let data = vec![0i8; rows * cols];
        let params = QuantizationParams::symmetric_per_tensor(1.0);
        return QuantizedWeight::new(data, vec![rows, cols], params);
    }

    let scale = max_val / 127.0;
    let (rows, cols) = array.dim();
    let mut data = Vec::with_capacity(rows * cols);

    for i in 0..rows {
        for j in 0..cols {
            let x = array[[i, j]];
            let q = (x / scale).round() as i32;
            let q_clamped = q.clamp(-128, 127) as i8;
            data.push(q_clamped);
        }
    }

    let params = QuantizationParams::symmetric_per_tensor(scale);
    QuantizedWeight::new(data, vec![rows, cols], params)
}

/// Quantize f32 2D array to INT8 using symmetric per-channel quantization
///
/// Each output channel (row) has independent scale factor
pub fn quantize_symmetric_per_channel(array: &Array2<f32>) -> ModelResult<QuantizedWeight> {
    let (rows, cols) = array.dim();
    let mut scales = Vec::with_capacity(rows);
    let mut data = Vec::with_capacity(rows * cols);

    // Compute scale per row (output channel)
    for i in 0..rows {
        let row = array.row(i);
        let max_val = row.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

        let scale = if max_val == 0.0 {
            1.0 // Avoid division by zero
        } else {
            max_val / 127.0
        };

        scales.push(scale);
    }

    // Quantize each row with its scale
    for i in 0..rows {
        let scale = scales[i];
        for j in 0..cols {
            let x = array[[i, j]];
            let q = (x / scale).round() as i32;
            let q_clamped = q.clamp(-128, 127) as i8;
            data.push(q_clamped);
        }
    }

    let params = QuantizationParams::symmetric_per_channel(scales);
    QuantizedWeight::new(data, vec![rows, cols], params)
}

/// Quantize f32 array to INT8 using asymmetric quantization
///
/// Computes optimal scale and zero-point from min/max values
pub fn quantize_asymmetric_1d(array: &Array1<f32>) -> ModelResult<QuantizedWeight> {
    let min_val = array.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = array.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if (max_val - min_val).abs() < 1e-8 {
        // Constant array
        let data = vec![0i8; array.len()];
        let params = QuantizationParams::asymmetric_per_tensor(1.0, 0);
        return QuantizedWeight::new(data, vec![array.len()], params);
    }

    let scale = (max_val - min_val) / 255.0;
    let zero_point_f = -128.0 - min_val / scale;
    let zero_point = zero_point_f.round().clamp(-128.0, 127.0) as i8;

    let mut data = Vec::with_capacity(array.len());

    for &x in array.iter() {
        let q_f = x / scale + zero_point as f32;
        let q = q_f.round().clamp(-128.0, 127.0) as i8;
        data.push(q);
    }

    let params = QuantizationParams::asymmetric_per_tensor(scale, zero_point);
    QuantizedWeight::new(data, vec![array.len()], params)
}

/// Statistics for quantization calibration
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    /// Minimum observed value
    pub min: f32,
    /// Maximum observed value
    pub max: f32,
    /// Number of observations
    pub count: usize,
}

impl CalibrationStats {
    /// Create new calibration stats
    pub fn new() -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            count: 0,
        }
    }

    /// Update statistics with new data
    pub fn update_1d(&mut self, data: &Array1<f32>) {
        for &x in data.iter() {
            self.min = self.min.min(x);
            self.max = self.max.max(x);
        }
        self.count += data.len();
    }

    /// Update statistics with 2D data
    pub fn update_2d(&mut self, data: &Array2<f32>) {
        for &x in data.iter() {
            self.min = self.min.min(x);
            self.max = self.max.max(x);
        }
        self.count += data.len();
    }

    /// Compute symmetric quantization params from statistics
    pub fn to_symmetric_params(&self) -> ModelResult<QuantizationParams> {
        let max_abs = self.max.abs().max(self.min.abs());
        if max_abs == 0.0 {
            Ok(QuantizationParams::symmetric_per_tensor(1.0))
        } else {
            Ok(QuantizationParams::symmetric_per_tensor(max_abs / 127.0))
        }
    }

    /// Compute asymmetric quantization params from statistics
    pub fn to_asymmetric_params(&self) -> ModelResult<QuantizationParams> {
        if (self.max - self.min).abs() < 1e-8 {
            Ok(QuantizationParams::asymmetric_per_tensor(1.0, 0))
        } else {
            let scale = (self.max - self.min) / 255.0;
            let zero_point_f = -128.0 - self.min / scale;
            let zero_point = zero_point_f.round().clamp(-128.0, 127.0) as i8;
            Ok(QuantizationParams::asymmetric_per_tensor(scale, zero_point))
        }
    }
}

impl Default for CalibrationStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Activation quantization for dynamic runtime quantization
///
/// Quantizes activations on-the-fly during inference for additional memory savings
#[derive(Debug, Clone)]
pub struct ActivationQuantizer {
    /// Method to use
    method: QuantizationMethod,
    /// Granularity
    #[allow(dead_code)]
    granularity: QuantizationGranularity,
    /// Calibration statistics (optional, for static quantization)
    calibration: Option<QuantizationParams>,
}

impl ActivationQuantizer {
    /// Create a new activation quantizer with symmetric per-tensor quantization
    pub fn new_symmetric() -> Self {
        Self {
            method: QuantizationMethod::Symmetric,
            granularity: QuantizationGranularity::PerTensor,
            calibration: None,
        }
    }

    /// Create a new activation quantizer with asymmetric per-tensor quantization
    pub fn new_asymmetric() -> Self {
        Self {
            method: QuantizationMethod::Asymmetric,
            granularity: QuantizationGranularity::PerTensor,
            calibration: None,
        }
    }

    /// Set calibration parameters from statistics
    pub fn calibrate(&mut self, stats: &CalibrationStats) -> ModelResult<()> {
        self.calibration = Some(match self.method {
            QuantizationMethod::Symmetric => stats.to_symmetric_params()?,
            QuantizationMethod::Asymmetric => stats.to_asymmetric_params()?,
        });
        Ok(())
    }

    /// Quantize activation (1D)
    pub fn quantize_activation_1d(&self, activation: &Array1<f32>) -> ModelResult<Vec<i8>> {
        let params = if let Some(ref cal) = self.calibration {
            // Use calibrated parameters
            cal.clone()
        } else {
            // Compute parameters on-the-fly
            let min_val = activation.iter().copied().fold(f32::INFINITY, f32::min);
            let max_val = activation.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            match self.method {
                QuantizationMethod::Symmetric => {
                    let max_abs = max_val.abs().max(min_val.abs());
                    QuantizationParams::symmetric_per_tensor(max_abs / 127.0)
                }
                QuantizationMethod::Asymmetric => {
                    let scale = (max_val - min_val) / 255.0;
                    let zero_point = (-128.0 - min_val / scale).round().clamp(-128.0, 127.0) as i8;
                    QuantizationParams::asymmetric_per_tensor(scale, zero_point)
                }
            }
        };

        let scale = params.scale[0];
        let zero_point = params.zero_point[0];

        let mut quantized = Vec::with_capacity(activation.len());
        for &x in activation.iter() {
            let q = match self.method {
                QuantizationMethod::Symmetric => (x / scale).round().clamp(-128.0, 127.0) as i8,
                QuantizationMethod::Asymmetric => {
                    let q_f = x / scale + zero_point as f32;
                    q_f.round().clamp(-128.0, 127.0) as i8
                }
            };
            quantized.push(q);
        }

        Ok(quantized)
    }

    /// Dequantize activation (1D)
    pub fn dequantize_activation_1d(
        &self,
        quantized: &[i8],
        original_len: usize,
    ) -> ModelResult<Array1<f32>> {
        if quantized.len() != original_len {
            return Err(ModelError::invalid_config(format!(
                "quantized length {} doesn't match expected {}",
                quantized.len(),
                original_len
            )));
        }

        let params = self
            .calibration
            .as_ref()
            .ok_or_else(|| ModelError::invalid_config("calibration required for dequantization"))?;

        let scale = params.scale[0];
        let zero_point = params.zero_point[0];

        let mut result = Array1::zeros(original_len);
        for (i, &q) in quantized.iter().enumerate() {
            result[i] = (q as i32 - zero_point as i32) as f32 * scale;
        }

        Ok(result)
    }

    /// Quantize and immediately dequantize (simulates quantization error)
    pub fn simulate_quantization(&self, activation: &Array1<f32>) -> ModelResult<Array1<f32>> {
        // Compute quantization parameters
        let min_val = activation.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = activation.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let (scale, zero_point) = match self.method {
            QuantizationMethod::Symmetric => {
                let max_abs = max_val.abs().max(min_val.abs());
                (max_abs / 127.0, 0)
            }
            QuantizationMethod::Asymmetric => {
                let scale = (max_val - min_val) / 255.0;
                let zp = (-128.0 - min_val / scale).round().clamp(-128.0, 127.0) as i8;
                (scale, zp)
            }
        };

        // Quantize and dequantize in one pass
        let mut result = Array1::zeros(activation.len());
        for (i, &x) in activation.iter().enumerate() {
            let q = match self.method {
                QuantizationMethod::Symmetric => (x / scale).round().clamp(-128.0, 127.0) as i8,
                QuantizationMethod::Asymmetric => {
                    let q_f = x / scale + zero_point as f32;
                    q_f.round().clamp(-128.0, 127.0) as i8
                }
            };
            result[i] = (q as i32 - zero_point as i32) as f32 * scale;
        }

        Ok(result)
    }

    /// Get memory savings estimate (percentage)
    pub fn memory_savings(&self) -> f32 {
        // INT8 uses 1 byte vs FP32's 4 bytes
        75.0 // 75% savings
    }
}

impl Default for ActivationQuantizer {
    fn default() -> Self {
        Self::new_symmetric()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_symmetric_quantization_1d() {
        let array = Array1::from_vec(vec![-10.0, -5.0, 0.0, 5.0, 10.0]);
        let quantized = quantize_symmetric_1d(&array).expect("Failed to quantize 1d array");

        assert_eq!(quantized.shape, vec![5]);
        assert_eq!(quantized.params.method, QuantizationMethod::Symmetric);

        // Dequantize and check error
        let dequantized = quantized
            .dequantize_1d()
            .expect("Failed to dequantize 1d array");
        for i in 0..5 {
            assert!(approx_eq(array[i], dequantized[i], 0.1));
        }
    }

    #[test]
    fn test_symmetric_quantization_2d() {
        let array = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
            .expect("Failed to create test array");
        let quantized = quantize_symmetric_2d(&array).expect("Failed to quantize 2d array");

        assert_eq!(quantized.shape, vec![2, 3]);

        let dequantized = quantized
            .dequantize_2d()
            .expect("Failed to dequantize 2d array");
        for i in 0..2 {
            for j in 0..3 {
                assert!(approx_eq(array[[i, j]], dequantized[[i, j]], 0.05));
            }
        }
    }

    #[test]
    fn test_per_channel_quantization() {
        // Different ranges per channel
        let array = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
            .expect("Failed to create test array");
        let quantized =
            quantize_symmetric_per_channel(&array).expect("Failed to quantize per channel");

        assert_eq!(
            quantized.params.granularity,
            QuantizationGranularity::PerChannel
        );
        assert_eq!(quantized.params.scale.len(), 2);

        let dequantized = quantized
            .dequantize_2d()
            .expect("Failed to dequantize 2d array");

        // Per-channel quantization should give better accuracy for different ranges
        for i in 0..2 {
            for j in 0..3 {
                assert!(approx_eq(array[[i, j]], dequantized[[i, j]], 0.3));
            }
        }
    }

    #[test]
    fn test_asymmetric_quantization() {
        let array = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let quantized = quantize_asymmetric_1d(&array).expect("Failed to quantize asymmetric");

        assert_eq!(quantized.params.method, QuantizationMethod::Asymmetric);

        let dequantized = quantized.dequantize_1d().expect("Failed to dequantize");
        for i in 0..5 {
            assert!(approx_eq(array[i], dequantized[i], 0.05));
        }
    }

    #[test]
    fn test_calibration_stats() {
        let mut stats = CalibrationStats::new();

        let data1 = Array1::from_vec(vec![-5.0, 0.0, 5.0]);
        let data2 = Array1::from_vec(vec![-10.0, -2.0, 8.0]);

        stats.update_1d(&data1);
        stats.update_1d(&data2);

        assert_eq!(stats.min, -10.0);
        assert_eq!(stats.max, 8.0);
        assert_eq!(stats.count, 6);

        let params = stats.to_symmetric_params().expect("Failed to get params");
        assert!(approx_eq(params.scale[0], 10.0 / 127.0, 1e-6));
    }

    #[test]
    fn test_memory_savings() {
        let array = Array2::from_shape_vec((100, 100), vec![1.0; 10000])
            .expect("Failed to create test array");
        let quantized = quantize_symmetric_2d(&array).expect("Failed to quantize");

        // INT8: 1 byte per element, FP32: 4 bytes per element
        let original_size = 10000 * 4;
        let quantized_size = quantized.memory_size();

        assert_eq!(quantized_size, 10000); // 1 byte per element
        assert!(quantized_size < original_size / 3); // At least 4x compression
    }

    #[test]
    fn test_activation_quantizer_symmetric() {
        let quantizer = ActivationQuantizer::new_symmetric();
        let activation = Array1::from_vec(vec![-10.0, -5.0, 0.0, 5.0, 10.0]);

        // Quantize
        let quantized = quantizer
            .quantize_activation_1d(&activation)
            .expect("Failed to quantize activation");
        assert_eq!(quantized.len(), activation.len());

        // Check memory savings
        assert_eq!(quantizer.memory_savings(), 75.0);
    }

    #[test]
    fn test_activation_quantizer_asymmetric() {
        let quantizer = ActivationQuantizer::new_asymmetric();
        let activation = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let quantized = quantizer
            .quantize_activation_1d(&activation)
            .expect("Failed to quantize activation");
        assert_eq!(quantized.len(), activation.len());
    }

    #[test]
    fn test_activation_quantizer_with_calibration() {
        let mut quantizer = ActivationQuantizer::new_symmetric();

        // Collect calibration statistics
        let mut stats = CalibrationStats::new();
        stats.update_1d(&Array1::from_vec(vec![-10.0, 0.0, 10.0]));
        stats.update_1d(&Array1::from_vec(vec![-5.0, 0.0, 5.0]));

        // Calibrate quantizer
        quantizer.calibrate(&stats).expect("Failed to calibrate");

        // Now quantize with calibrated parameters
        let activation = Array1::from_vec(vec![-8.0, 0.0, 8.0]);
        let quantized = quantizer
            .quantize_activation_1d(&activation)
            .expect("Failed to quantize activation");

        // Dequantize
        let dequantized = quantizer
            .dequantize_activation_1d(&quantized, activation.len())
            .expect("Failed to dequantize activation");

        // Check accuracy
        for i in 0..activation.len() {
            assert!((activation[i] - dequantized[i]).abs() < 1.0);
        }
    }

    #[test]
    fn test_simulate_quantization() {
        let quantizer = ActivationQuantizer::new_symmetric();
        let activation = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let simulated = quantizer
            .simulate_quantization(&activation)
            .expect("Failed to simulate quantization");

        // Simulated values should be close to original
        for i in 0..activation.len() {
            assert!((activation[i] - simulated[i]).abs() < 0.1);
        }
    }
}
