//! # Dynamic Quantization
//!
//! Post-training quantization for model compression and faster inference.
//!
//! ## Features
//!
//! - **INT8 Quantization**: 8-bit integer quantization with calibration
//! - **INT4 Quantization**: 4-bit quantization for extreme compression
//! - **Per-Tensor Quantization**: Single scale/zero-point per tensor
//! - **Per-Channel Quantization**: Separate scale/zero-point per channel
//! - **Dynamic Range**: Automatic dynamic range calculation
//! - **Calibration**: Statistics collection for better quantization
//!
//! ## References
//!
//! - "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
//! - "ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers"

use crate::{CoreError, CoreResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Quantization data type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// 8-bit signed integer
    INT8,
    /// 4-bit signed integer
    INT4,
    /// 16-bit floating point
    FP16,
}

/// Quantization scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// Per-tensor quantization (single scale and zero-point)
    PerTensor,
    /// Per-channel quantization (separate scale and zero-point per output channel)
    PerChannel,
}

/// Quantization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Quantization type
    pub qtype: QuantizationType,
    /// Quantization scheme
    pub scheme: QuantizationScheme,
    /// Scale factors (per-tensor or per-channel)
    pub scales: Vec<f32>,
    /// Zero points (per-tensor or per-channel)
    pub zero_points: Vec<i32>,
    /// Original shape
    pub shape: Vec<usize>,
}

impl QuantizationParams {
    /// Create new quantization parameters
    pub fn new(
        qtype: QuantizationType,
        scheme: QuantizationScheme,
        scales: Vec<f32>,
        zero_points: Vec<i32>,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            qtype,
            scheme,
            scales,
            zero_points,
            shape,
        }
    }

    /// Get quantization range
    pub fn qrange(&self) -> (i32, i32) {
        match self.qtype {
            QuantizationType::INT8 => (-128, 127),
            QuantizationType::INT4 => (-8, 7),
            QuantizationType::FP16 => (0, 0), // Not applicable for FP16
        }
    }

    /// Validate parameters
    pub fn validate(&self) -> CoreResult<()> {
        match self.scheme {
            QuantizationScheme::PerTensor => {
                if self.scales.len() != 1 || self.zero_points.len() != 1 {
                    return Err(CoreError::InvalidConfig(
                        "PerTensor scheme requires exactly 1 scale and zero-point".into(),
                    ));
                }
            }
            QuantizationScheme::PerChannel => {
                if self.shape.is_empty() {
                    return Err(CoreError::InvalidConfig(
                        "PerChannel scheme requires shape information".into(),
                    ));
                }
                let num_channels = self.shape[0];
                if self.scales.len() != num_channels || self.zero_points.len() != num_channels {
                    return Err(CoreError::InvalidConfig(format!(
                        "PerChannel scheme requires {} scales and zero-points, got {} and {}",
                        num_channels,
                        self.scales.len(),
                        self.zero_points.len()
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Quantized tensor representation
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data (stored as i8 or i16)
    pub data: Vec<i8>,
    /// Quantization parameters
    pub params: QuantizationParams,
}

impl QuantizedTensor {
    /// Create a new quantized tensor
    pub fn new(data: Vec<i8>, params: QuantizationParams) -> CoreResult<Self> {
        params.validate()?;
        Ok(Self { data, params })
    }

    /// Dequantize to f32 array
    pub fn dequantize_1d(&self) -> CoreResult<Array1<f32>> {
        if self.params.shape.len() != 1 {
            return Err(CoreError::InvalidConfig(
                "Expected 1D tensor for dequantize_1d".into(),
            ));
        }

        let size = self.params.shape[0];
        let mut result = Array1::zeros(size);

        match self.params.scheme {
            QuantizationScheme::PerTensor => {
                let scale = self.params.scales[0];
                let zero_point = self.params.zero_points[0];

                for (i, &q_val) in self.data.iter().enumerate() {
                    result[i] = (q_val as i32 - zero_point) as f32 * scale;
                }
            }
            QuantizationScheme::PerChannel => {
                // For 1D, per-channel doesn't make sense, treat as per-tensor
                let scale = self.params.scales[0];
                let zero_point = self.params.zero_points[0];

                for (i, &q_val) in self.data.iter().enumerate() {
                    result[i] = (q_val as i32 - zero_point) as f32 * scale;
                }
            }
        }

        Ok(result)
    }

    /// Dequantize to f32 2D array
    pub fn dequantize_2d(&self) -> CoreResult<Array2<f32>> {
        if self.params.shape.len() != 2 {
            return Err(CoreError::InvalidConfig(
                "Expected 2D tensor for dequantize_2d".into(),
            ));
        }

        let rows = self.params.shape[0];
        let cols = self.params.shape[1];
        let mut result = Array2::zeros((rows, cols));

        match self.params.scheme {
            QuantizationScheme::PerTensor => {
                let scale = self.params.scales[0];
                let zero_point = self.params.zero_points[0];

                for i in 0..rows {
                    for j in 0..cols {
                        let idx = i * cols + j;
                        let q_val = self.data[idx];
                        result[[i, j]] = (q_val as i32 - zero_point) as f32 * scale;
                    }
                }
            }
            QuantizationScheme::PerChannel => {
                // Per-channel: one scale/zero-point per output channel (row)
                for i in 0..rows {
                    let scale = self.params.scales[i];
                    let zero_point = self.params.zero_points[i];

                    for j in 0..cols {
                        let idx = i * cols + j;
                        let q_val = self.data[idx];
                        result[[i, j]] = (q_val as i32 - zero_point) as f32 * scale;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.data.len() * std::mem::size_of::<f32>();
        let quantized_size = self.data.len() * std::mem::size_of::<i8>()
            + self.params.scales.len() * std::mem::size_of::<f32>()
            + self.params.zero_points.len() * std::mem::size_of::<i32>();
        original_size as f32 / quantized_size as f32
    }
}

/// Dynamic quantizer
pub struct DynamicQuantizer {
    /// Quantization type
    qtype: QuantizationType,
    /// Quantization scheme
    scheme: QuantizationScheme,
}

impl DynamicQuantizer {
    /// Create a new dynamic quantizer
    pub fn new(qtype: QuantizationType, scheme: QuantizationScheme) -> Self {
        Self { qtype, scheme }
    }

    /// Create INT8 per-tensor quantizer
    pub fn int8_per_tensor() -> Self {
        Self::new(QuantizationType::INT8, QuantizationScheme::PerTensor)
    }

    /// Create INT8 per-channel quantizer
    pub fn int8_per_channel() -> Self {
        Self::new(QuantizationType::INT8, QuantizationScheme::PerChannel)
    }

    /// Create INT4 per-channel quantizer
    pub fn int4_per_channel() -> Self {
        Self::new(QuantizationType::INT4, QuantizationScheme::PerChannel)
    }

    /// Quantize a 1D array
    pub fn quantize_1d(&self, data: &Array1<f32>) -> CoreResult<QuantizedTensor> {
        let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let (qmin, qmax) = self.get_qrange();

        // Handle case where all values are the same (avoid division by zero)
        let scale = if (max_val - min_val).abs() < 1e-8 {
            1.0
        } else {
            (max_val - min_val) / (qmax - qmin) as f32
        };

        let zero_point = if (max_val - min_val).abs() < 1e-8 {
            0
        } else {
            qmin - (min_val / scale).round() as i32
        };

        let mut quantized = Vec::with_capacity(data.len());
        for &val in data.iter() {
            let q_val = (val / scale).round() as i32 + zero_point;
            let q_val_clamped = q_val.clamp(qmin, qmax);
            quantized.push(q_val_clamped as i8);
        }

        let params = QuantizationParams::new(
            self.qtype,
            self.scheme,
            vec![scale],
            vec![zero_point],
            vec![data.len()],
        );

        QuantizedTensor::new(quantized, params)
    }

    /// Quantize a 2D array
    pub fn quantize_2d(&self, data: &Array2<f32>) -> CoreResult<QuantizedTensor> {
        let (rows, cols) = data.dim();
        let (qmin, qmax) = self.get_qrange();

        match self.scheme {
            QuantizationScheme::PerTensor => {
                // Find global min/max
                let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                let scale = (max_val - min_val) / (qmax - qmin) as f32;
                let zero_point = qmin - (min_val / scale).round() as i32;

                let mut quantized = Vec::with_capacity(rows * cols);
                for &val in data.iter() {
                    let q_val = (val / scale).round() as i32 + zero_point;
                    let q_val_clamped = q_val.clamp(qmin, qmax);
                    quantized.push(q_val_clamped as i8);
                }

                let params = QuantizationParams::new(
                    self.qtype,
                    self.scheme,
                    vec![scale],
                    vec![zero_point],
                    vec![rows, cols],
                );

                QuantizedTensor::new(quantized, params)
            }
            QuantizationScheme::PerChannel => {
                // Per-channel: compute scale/zero-point for each row
                let mut scales = Vec::with_capacity(rows);
                let mut zero_points = Vec::with_capacity(rows);
                let mut quantized = Vec::with_capacity(rows * cols);

                for row in data.axis_iter(Axis(0)) {
                    let min_val = row.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                    let scale = (max_val - min_val) / (qmax - qmin) as f32;
                    let zero_point = qmin - (min_val / scale).round() as i32;

                    scales.push(scale);
                    zero_points.push(zero_point);

                    for &val in row.iter() {
                        let q_val = (val / scale).round() as i32 + zero_point;
                        let q_val_clamped = q_val.clamp(qmin, qmax);
                        quantized.push(q_val_clamped as i8);
                    }
                }

                let params = QuantizationParams::new(
                    self.qtype,
                    self.scheme,
                    scales,
                    zero_points,
                    vec![rows, cols],
                );

                QuantizedTensor::new(quantized, params)
            }
        }
    }

    /// Get quantization range
    fn get_qrange(&self) -> (i32, i32) {
        match self.qtype {
            QuantizationType::INT8 => (-128, 127),
            QuantizationType::INT4 => (-8, 7),
            QuantizationType::FP16 => (0, 0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_types() {
        let qt = QuantizationType::INT8;
        assert_eq!(qt, QuantizationType::INT8);

        let qs = QuantizationScheme::PerTensor;
        assert_eq!(qs, QuantizationScheme::PerTensor);
    }

    #[test]
    fn test_quantization_params() {
        let params = QuantizationParams::new(
            QuantizationType::INT8,
            QuantizationScheme::PerTensor,
            vec![0.1],
            vec![0],
            vec![100],
        );

        assert_eq!(params.qtype, QuantizationType::INT8);
        assert_eq!(params.qrange(), (-128, 127));
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_params_validation() {
        // PerTensor with wrong number of scales
        let mut params = QuantizationParams::new(
            QuantizationType::INT8,
            QuantizationScheme::PerTensor,
            vec![0.1, 0.2],
            vec![0],
            vec![100],
        );
        assert!(params.validate().is_err());

        // PerChannel with wrong number of scales
        params = QuantizationParams::new(
            QuantizationType::INT8,
            QuantizationScheme::PerChannel,
            vec![0.1],
            vec![0, 1],
            vec![2, 100],
        );
        assert!(params.validate().is_err());

        // Correct PerChannel
        params = QuantizationParams::new(
            QuantizationType::INT8,
            QuantizationScheme::PerChannel,
            vec![0.1, 0.2],
            vec![0, 1],
            vec![2, 100],
        );
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_dynamic_quantizer_creation() {
        let quantizer = DynamicQuantizer::int8_per_tensor();
        assert_eq!(quantizer.qtype, QuantizationType::INT8);
        assert_eq!(quantizer.scheme, QuantizationScheme::PerTensor);

        let quantizer = DynamicQuantizer::int4_per_channel();
        assert_eq!(quantizer.qtype, QuantizationType::INT4);
        assert_eq!(quantizer.scheme, QuantizationScheme::PerChannel);
    }

    #[test]
    fn test_quantize_dequantize_1d() {
        let quantizer = DynamicQuantizer::int8_per_tensor();
        let data = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let quantized = quantizer.quantize_1d(&data).unwrap();
        assert_eq!(quantized.data.len(), 5);

        let dequantized = quantized.dequantize_1d().unwrap();
        assert_eq!(dequantized.len(), 5);

        // Check approximate reconstruction
        for i in 0..5 {
            let error = (dequantized[i] - data[i]).abs();
            assert!(error < 0.1, "Reconstruction error too large: {}", error);
        }
    }

    #[test]
    fn test_quantize_dequantize_2d() {
        let quantizer = DynamicQuantizer::int8_per_tensor();
        let data = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32);

        let quantized = quantizer.quantize_2d(&data).unwrap();
        assert_eq!(quantized.data.len(), 16);

        let dequantized = quantized.dequantize_2d().unwrap();
        assert_eq!(dequantized.shape(), &[4, 4]);

        // Check approximate reconstruction
        for i in 0..4 {
            for j in 0..4 {
                let error = (dequantized[[i, j]] - data[[i, j]]).abs();
                assert!(error < 0.5, "Reconstruction error too large: {}", error);
            }
        }
    }

    #[test]
    fn test_per_channel_quantization() {
        let quantizer = DynamicQuantizer::int8_per_channel();
        let data = Array2::from_shape_fn((3, 4), |(i, j)| (i * 10 + j) as f32);

        let quantized = quantizer.quantize_2d(&data).unwrap();
        assert_eq!(quantized.params.scales.len(), 3); // One per channel (row)
        assert_eq!(quantized.params.zero_points.len(), 3);

        let dequantized = quantized.dequantize_2d().unwrap();
        assert_eq!(dequantized.shape(), &[3, 4]);

        // Check reconstruction
        for i in 0..3 {
            for j in 0..4 {
                let error = (dequantized[[i, j]] - data[[i, j]]).abs();
                assert!(error < 1.0, "Error at [{}, {}]: {}", i, j, error);
            }
        }
    }

    #[test]
    fn test_compression_ratio() {
        let quantizer = DynamicQuantizer::int8_per_tensor();
        let data = Array2::from_shape_fn((100, 100), |(i, j)| (i + j) as f32);

        let quantized = quantizer.quantize_2d(&data).unwrap();
        let ratio = quantized.compression_ratio();

        // INT8 should give ~4x compression (32-bit float -> 8-bit int)
        // With overhead for scale/zero-point, expect ~3.9x
        assert!(
            ratio > 3.5 && ratio < 4.1,
            "Unexpected compression ratio: {}",
            ratio
        );
    }

    #[test]
    fn test_qrange() {
        let quantizer = DynamicQuantizer::int8_per_tensor();
        assert_eq!(quantizer.get_qrange(), (-128, 127));

        let quantizer = DynamicQuantizer::int4_per_channel();
        assert_eq!(quantizer.get_qrange(), (-8, 7));
    }

    #[test]
    fn test_extreme_values() {
        let quantizer = DynamicQuantizer::int8_per_tensor();
        let data = Array1::from_vec(vec![-100.0, -50.0, 0.0, 50.0, 100.0]);

        let quantized = quantizer.quantize_1d(&data).unwrap();
        let dequantized = quantized.dequantize_1d().unwrap();

        // Extreme values should be preserved reasonably well
        for i in 0..5 {
            let error_pct = ((dequantized[i] - data[i]) / data[i].abs().max(1.0)).abs();
            assert!(
                error_pct < 0.05,
                "Large error at index {}: {}%",
                i,
                error_pct * 100.0
            );
        }
    }
}
