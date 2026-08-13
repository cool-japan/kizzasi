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
    /// Quantized data. Layout depends on `params.qtype`:
    /// - [`QuantizationType::INT8`]: one `i8` per logical element
    ///   (`data.len() == params.shape.iter().product()`).
    /// - [`QuantizationType::INT4`]: two 4-bit signed nibbles packed per
    ///   byte, low nibble first (`data.len() ==
    ///   params.shape.iter().product::<usize>().div_ceil(2)`). Use
    ///   `pack_int4`/`unpack_int4` rather than indexing `data` directly.
    /// - [`QuantizationType::FP16`]: not produced by [`DynamicQuantizer`]
    ///   (rejected at quantize time -- see its docs).
    pub data: Vec<i8>,
    /// Quantization parameters
    pub params: QuantizationParams,
}

/// Pack signed 4-bit values (each already clamped to `[-8, 7]`) two per
/// byte, low nibble first. The final byte's high nibble is zero-filled when
/// `values.len()` is odd (never read back, since unpacking is always driven
/// by the original logical element count).
fn pack_int4(values: &[i8]) -> Vec<i8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(2));
    let mut pair = values.chunks(2);
    for chunk in &mut pair {
        let lo = (chunk[0] as u8) & 0x0F;
        let hi = if chunk.len() == 2 {
            (chunk[1] as u8) & 0x0F
        } else {
            0
        };
        packed.push(((hi << 4) | lo) as i8);
    }
    packed
}

/// Unpack `count` signed 4-bit values (two per byte, low nibble first),
/// sign-extending each nibble back to `i32`.
fn unpack_int4(packed: &[i8], count: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let byte_idx = i / 2;
        let byte = packed.get(byte_idx).copied().unwrap_or(0) as u8;
        let nibble = if i % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
        // Sign-extend the 4-bit two's-complement nibble to i32.
        let signed = if nibble & 0x08 != 0 {
            (nibble as i32) - 16
        } else {
            nibble as i32
        };
        out.push(signed);
    }
    out
}

impl QuantizedTensor {
    /// Create a new quantized tensor
    pub fn new(data: Vec<i8>, params: QuantizationParams) -> CoreResult<Self> {
        params.validate()?;
        Ok(Self { data, params })
    }

    /// Unpack `self.data` into one `i32` per logical element, transparently
    /// handling [`QuantizationType::INT4`]'s two-nibbles-per-byte packing.
    fn unpacked_q_values(&self, count: usize) -> Vec<i32> {
        if self.params.qtype == QuantizationType::INT4 {
            unpack_int4(&self.data, count)
        } else {
            self.data.iter().map(|&v| v as i32).collect()
        }
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
        let q_values = self.unpacked_q_values(size);

        match self.params.scheme {
            QuantizationScheme::PerTensor => {
                let scale = self.params.scales[0];
                let zero_point = self.params.zero_points[0];

                for (i, &q_val) in q_values.iter().enumerate() {
                    result[i] = (q_val - zero_point) as f32 * scale;
                }
            }
            QuantizationScheme::PerChannel => {
                // For 1D, per-channel doesn't make sense, treat as per-tensor
                let scale = self.params.scales[0];
                let zero_point = self.params.zero_points[0];

                for (i, &q_val) in q_values.iter().enumerate() {
                    result[i] = (q_val - zero_point) as f32 * scale;
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
        let q_values = self.unpacked_q_values(rows * cols);

        match self.params.scheme {
            QuantizationScheme::PerTensor => {
                let scale = self.params.scales[0];
                let zero_point = self.params.zero_points[0];

                for i in 0..rows {
                    for j in 0..cols {
                        let idx = i * cols + j;
                        result[[i, j]] = (q_values[idx] - zero_point) as f32 * scale;
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
                        result[[i, j]] = (q_values[idx] - zero_point) as f32 * scale;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Get compression ratio versus a dense `f32` tensor of the same shape.
    ///
    /// Computed from the *logical* element count (`params.shape`) and the
    /// actual on-disk bit width of `params.qtype`, not from `data.len()`
    /// directly -- for [`QuantizationType::INT4`], `data.len()` is the
    /// *packed* byte count (`ceil(n/2)`), which is half the element count,
    /// not equal to it.
    pub fn compression_ratio(&self) -> f32 {
        let num_elements: usize = self.params.shape.iter().product();
        let original_size = num_elements * std::mem::size_of::<f32>();

        let quantized_data_bytes = match self.params.qtype {
            QuantizationType::INT4 => num_elements.div_ceil(2), // 2 values/byte
            QuantizationType::INT8 => num_elements,             // 1 value/byte
            QuantizationType::FP16 => num_elements * 2, // not produced today; kept exhaustive
        };
        let quantized_size = quantized_data_bytes
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

    /// Reject [`QuantizationType::FP16`], which `DynamicQuantizer` does not
    /// implement (`get_qrange()` returns `(0, 0)` for it, which would make
    /// `quantize_1d`/`quantize_2d` compute `scale = range / 0 = inf`,
    /// silently zero every quantized value, and turn every dequantized
    /// value into `NaN` via `0 * inf`). A real FP16 path would need to widen
    /// [`QuantizedTensor::data`] beyond `Vec<i8>` to store `half::f16`
    /// payloads; until that lands, fail loudly instead of producing
    /// silently-corrupt output.
    fn reject_unsupported_qtype(&self) -> CoreResult<()> {
        if self.qtype == QuantizationType::FP16 {
            return Err(CoreError::InvalidConfig(
                "DynamicQuantizer: QuantizationType::FP16 is not implemented (quantize_1d/quantize_2d would silently produce zeros/NaN); use INT8 or INT4".to_string(),
            ));
        }
        Ok(())
    }

    /// Quantize a 1D array
    pub fn quantize_1d(&self, data: &Array1<f32>) -> CoreResult<QuantizedTensor> {
        self.reject_unsupported_qtype()?;

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
            // NOT clamped to [qmin, qmax]: `zero_point` is an offset applied
            // before the per-value clamp below, and for a range that sits
            // far from zero (e.g. [10, 13]) it legitimately falls outside
            // [qmin, qmax] -- clamping it here would shift every quantized
            // value by the clamped amount and corrupt the whole tensor. Only
            // the final `q_val` (after adding `zero_point`) needs clamping.
            qmin - (min_val / scale).round() as i32
        };

        let mut quantized = Vec::with_capacity(data.len());
        for &val in data.iter() {
            let q_val = (val / scale).round() as i32 + zero_point;
            let q_val_clamped = q_val.clamp(qmin, qmax);
            quantized.push(q_val_clamped as i8);
        }

        let stored_data = if self.qtype == QuantizationType::INT4 {
            pack_int4(&quantized)
        } else {
            quantized
        };

        let params = QuantizationParams::new(
            self.qtype,
            self.scheme,
            vec![scale],
            vec![zero_point],
            vec![data.len()],
        );

        QuantizedTensor::new(stored_data, params)
    }

    /// Quantize a 2D array
    pub fn quantize_2d(&self, data: &Array2<f32>) -> CoreResult<QuantizedTensor> {
        self.reject_unsupported_qtype()?;

        let (rows, cols) = data.dim();
        let (qmin, qmax) = self.get_qrange();

        match self.scheme {
            QuantizationScheme::PerTensor => {
                // Find global min/max
                let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                // Degenerate-range guard (constant tensor / single value):
                // without this, `scale` is 0, `min_val / scale` is +-inf or
                // NaN, and `(inf).round() as i32` saturates to `i32::MAX`,
                // which then overflows the `qmin - i32::MAX` subtraction
                // below (panics in debug, wraps to garbage in release).
                // Mirrors the guard `quantize_1d` already has.
                let degenerate = (max_val - min_val).abs() < 1e-8;
                let scale = if degenerate {
                    1.0
                } else {
                    (max_val - min_val) / (qmax - qmin) as f32
                };
                // `zero_point` is intentionally NOT clamped to [qmin, qmax]
                // here -- see `quantize_1d` for why; only `q_val` below is.
                let zero_point = if degenerate {
                    0
                } else {
                    qmin - (min_val / scale).round() as i32
                };

                let mut quantized = Vec::with_capacity(rows * cols);
                for &val in data.iter() {
                    let q_val = (val / scale).round() as i32 + zero_point;
                    let q_val_clamped = q_val.clamp(qmin, qmax);
                    quantized.push(q_val_clamped as i8);
                }

                let stored_data = if self.qtype == QuantizationType::INT4 {
                    pack_int4(&quantized)
                } else {
                    quantized
                };

                let params = QuantizationParams::new(
                    self.qtype,
                    self.scheme,
                    vec![scale],
                    vec![zero_point],
                    vec![rows, cols],
                );

                QuantizedTensor::new(stored_data, params)
            }
            QuantizationScheme::PerChannel => {
                // Per-channel: compute scale/zero-point for each row
                let mut scales = Vec::with_capacity(rows);
                let mut zero_points = Vec::with_capacity(rows);
                let mut quantized = Vec::with_capacity(rows * cols);

                for row in data.axis_iter(Axis(0)) {
                    let min_val = row.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                    // Same degenerate-range guard as the PerTensor branch
                    // above, applied per row (e.g. a bias row of all-ones,
                    // or any all-identical channel).
                    let degenerate = (max_val - min_val).abs() < 1e-8;
                    let scale = if degenerate {
                        1.0
                    } else {
                        (max_val - min_val) / (qmax - qmin) as f32
                    };
                    // `zero_point` is intentionally NOT clamped to
                    // [qmin, qmax] here -- see `quantize_1d` for why; only
                    // `q_val` below is.
                    let zero_point = if degenerate {
                        0
                    } else {
                        qmin - (min_val / scale).round() as i32
                    };

                    scales.push(scale);
                    zero_points.push(zero_point);

                    for &val in row.iter() {
                        let q_val = (val / scale).round() as i32 + zero_point;
                        let q_val_clamped = q_val.clamp(qmin, qmax);
                        quantized.push(q_val_clamped as i8);
                    }
                }

                let stored_data = if self.qtype == QuantizationType::INT4 {
                    pack_int4(&quantized)
                } else {
                    quantized
                };

                let params = QuantizationParams::new(
                    self.qtype,
                    self.scheme,
                    scales,
                    zero_points,
                    vec![rows, cols],
                );

                QuantizedTensor::new(stored_data, params)
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

    // ------------------------------------------------------------------
    // Regression tests: quantize_2d zero-range guard, FP16 rejection, INT4
    // bit-packing.
    // ------------------------------------------------------------------

    #[test]
    fn test_quantize_2d_all_zeros_per_tensor_does_not_panic_or_nan() {
        // Regression: PerTensor quantize_2d had no degenerate-range guard.
        // For an all-zeros tensor, `min_val/scale` was NaN, `NaN as i32` is
        // 0, so quantization didn't panic here -- but exercise it anyway as
        // the base case for the guard.
        let quantizer = DynamicQuantizer::int8_per_tensor();
        let data = Array2::<f32>::zeros((4, 4));

        let quantized = quantizer.quantize_2d(&data).unwrap();
        let dequantized = quantized.dequantize_2d().unwrap();

        for v in dequantized.iter() {
            assert!(v.is_finite(), "expected finite value, got {v}");
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_quantize_2d_all_ones_per_tensor_does_not_panic_or_nan() {
        // Regression: this is the case that used to panic (debug) / produce
        // garbage (release). `Array2::from_elem((2,3), 1.0f32)`: scale would
        // be 0, `min_val/scale = inf`, `inf.round() as i32` saturates to
        // `i32::MAX`, and `qmin - i32::MAX` overflows i32.
        let quantizer = DynamicQuantizer::int8_per_tensor();
        let data = Array2::from_elem((2, 3), 1.0f32);

        let quantized = quantizer.quantize_2d(&data).unwrap();
        let dequantized = quantized.dequantize_2d().unwrap();

        for v in dequantized.iter() {
            assert!(v.is_finite(), "expected finite value, got {v}");
            assert!((v - 1.0).abs() < 1e-3, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_quantize_2d_all_ones_per_channel_does_not_panic_or_nan() {
        let quantizer = DynamicQuantizer::int8_per_channel();
        // Row 0 is constant (degenerate range); row 1 has real spread, so
        // this also checks the guard doesn't corrupt the non-degenerate row.
        let data = Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 1.0, 0.0, 5.0, 10.0]).unwrap();

        let quantized = quantizer.quantize_2d(&data).unwrap();
        let dequantized = quantized.dequantize_2d().unwrap();

        for j in 0..3 {
            assert!(dequantized[[0, j]].is_finite());
            assert!((dequantized[[0, j]] - 1.0).abs() < 1e-3);
        }
        for j in 0..3 {
            assert!(dequantized[[1, j]].is_finite());
            assert!((dequantized[[1, j]] - data[[1, j]]).abs() < 0.5);
        }
    }

    #[test]
    fn test_quantize_fp16_is_rejected() {
        // Regression: FP16 used to silently zero every value on quantize and
        // produce NaN on dequantize, with no error at any point.
        let quantizer =
            DynamicQuantizer::new(QuantizationType::FP16, QuantizationScheme::PerTensor);
        let data_1d = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let data_2d = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        assert!(quantizer.quantize_1d(&data_1d).is_err());
        assert!(quantizer.quantize_2d(&data_2d).is_err());

        let quantizer_pc =
            DynamicQuantizer::new(QuantizationType::FP16, QuantizationScheme::PerChannel);
        assert!(quantizer_pc.quantize_2d(&data_2d).is_err());
    }

    #[test]
    fn test_int4_data_is_bit_packed_two_per_byte() {
        let quantizer = DynamicQuantizer::int4_per_channel();
        // Odd element count so the packing's "half-empty last byte" path is
        // exercised too.
        let data = Array2::from_shape_fn((2, 5), |(i, j)| (i * 5 + j) as f32 * 0.1);

        let quantized = quantizer.quantize_2d(&data).unwrap();
        // 10 logical elements -> ceil(10/2) = 5 packed bytes. Before the
        // fix, this stored one full `i8` per element (10 bytes) -- zero
        // savings over INT8.
        assert_eq!(
            quantized.data.len(),
            10usize.div_ceil(2),
            "INT4 storage must be bit-packed two values per byte"
        );
    }

    #[test]
    fn test_int4_roundtrip_odd_and_even_lengths() {
        for n in [1usize, 2, 3, 7, 8, 15] {
            let quantizer = DynamicQuantizer::int4_per_channel();
            let data = Array2::from_shape_fn((1, n), |(_, j)| (j as f32) - (n as f32 / 2.0));

            let quantized = quantizer.quantize_2d(&data).unwrap();
            assert_eq!(quantized.data.len(), n.div_ceil(2));

            let dequantized = quantized.dequantize_2d().unwrap();
            assert_eq!(dequantized.dim(), (1, n));

            // INT4 has only 16 levels; round-trip error must stay within a
            // couple of quantization steps (the actual scale this tensor
            // was quantized with), proving the nibble packing/unpacking
            // round-trips correctly rather than checking exact precision.
            let scale = quantized.params.scales[0];
            let tolerance = 2.0 * scale + 1e-4;
            for j in 0..n {
                let err = (dequantized[[0, j]] - data[[0, j]]).abs();
                assert!(
                    err < tolerance,
                    "n={n} idx={j}: dequantized {} too far from original {} (scale={scale}, tol={tolerance})",
                    dequantized[[0, j]],
                    data[[0, j]]
                );
            }
        }
    }

    #[test]
    fn test_int4_compression_ratio_is_roughly_double_int8() {
        let data = Array2::from_shape_fn((64, 64), |(i, j)| ((i + j) as f32) * 0.01);

        let int8 = DynamicQuantizer::int8_per_tensor()
            .quantize_2d(&data)
            .unwrap();
        let int4 = DynamicQuantizer::new(QuantizationType::INT4, QuantizationScheme::PerTensor)
            .quantize_2d(&data)
            .unwrap();

        let ratio8 = int8.compression_ratio();
        let ratio4 = int4.compression_ratio();

        // Regression: compression_ratio() used to compute quantized size
        // from `data.len() * size_of::<i8>()` unconditionally, so INT4
        // reported the same ~4x ratio as INT8 despite storing half the
        // bytes. It should now be close to 2x INT8's ratio (ignoring the
        // small, shared scale/zero_point overhead).
        assert!(
            ratio4 > ratio8 * 1.5,
            "INT4 compression ratio ({ratio4}) should be substantially higher than INT8's ({ratio8})"
        );
    }
}
