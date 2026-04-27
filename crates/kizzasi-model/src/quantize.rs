//! Post-Training Quantization (PTQ) for kizzasi-model
//!
//! Provides INT8 (symmetric per-tensor or per-channel) and FP16 (half-precision)
//! quantization of weight maps, plus calibration-data tracking and round-trip
//! dequantization for accuracy evaluation.
//!
//! # Design
//!
//! The module is weight-map oriented: the primary input is a
//! `HashMap<String, Vec<f32>>` (tensor name → flat f32 values) and the primary
//! output is `HashMap<String, QuantizedTensor>`.  This is complementary to the
//! existing `quantization.rs`, which operates on `ndarray` types.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use kizzasi_model::quantize::{quantize_to_int8, ModelQuantizer};
//! use std::collections::HashMap;
//!
//! let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
//! weights.insert("embed".to_string(), vec![0.1, -0.2, 0.3]);
//!
//! // Convenience wrapper
//! let quantized = quantize_to_int8(&weights).unwrap();
//!
//! // Round-trip
//! let recovered = ModelQuantizer::dequantize_weights(&quantized).unwrap();
//! ```

use crate::error::{ModelError, ModelResult};
use half::f16;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Quantization precision mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantMode {
    /// Symmetric per-tensor INT8 quantization (values in −127..127).
    Int8,
    /// IEEE 754 half-precision (FP16).
    Fp16,
    /// Identity: no quantization (f32 pass-through).
    Fp32,
}

/// Configuration passed to [`ModelQuantizer`].
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Which numeric format to use.
    pub mode: QuantMode,
    /// When `true`, compute one scale per output-channel (dim 0) for INT8;
    /// when `false`, use a single per-tensor scale.
    pub per_channel: bool,
    /// When `true`, use symmetric quantization (zero_point = 0).
    pub symmetric: bool,
    /// Number of calibration batches to collect before computing scales.
    pub calibration_samples: usize,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            mode: QuantMode::Int8,
            per_channel: false,
            symmetric: true,
            calibration_samples: 64,
        }
    }
}

/// A quantized tensor: holds scale(s), zero-point(s), shape, and compressed data.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Original weight name (for logging / round-trip matching).
    pub name: String,
    /// Flat tensor shape (row-major).
    pub shape: Vec<usize>,
    /// Precision mode used during quantization.
    pub mode: QuantMode,
    /// Per-channel or per-tensor scale factor(s).
    pub scale: Vec<f32>,
    /// Per-channel or per-tensor zero-point(s).
    pub zero_point: Vec<i32>,
    /// INT8 quantized data (present when `mode == QuantMode::Int8`).
    pub data_i8: Option<Vec<i8>>,
    /// FP16 quantized data as raw `u16` bits (present when `mode == QuantMode::Fp16`).
    pub data_f16: Option<Vec<u16>>,
}

impl QuantizedTensor {
    /// Dequantize back to a flat `Vec<f32>`.
    ///
    /// For `Fp32` mode the function returns an error because no compressed data
    /// is stored; use the original weights directly instead.
    pub fn dequantize(&self) -> ModelResult<Vec<f32>> {
        match self.mode {
            QuantMode::Int8 => self.dequantize_int8(),
            QuantMode::Fp16 => self.dequantize_fp16(),
            QuantMode::Fp32 => Err(ModelError::quantization_error(
                "Fp32 mode stores no compressed data; dequantize is a no-op",
            )),
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn dequantize_int8(&self) -> ModelResult<Vec<f32>> {
        let data = self.data_i8.as_ref().ok_or_else(|| {
            ModelError::quantization_error("INT8 tensor is missing data_i8 buffer")
        })?;

        let num_elements = data.len();
        let num_channels = self.scale.len(); // 1 for per-tensor

        if self.zero_point.len() != num_channels {
            return Err(ModelError::quantization_error(format!(
                "scale length ({}) != zero_point length ({})",
                num_channels,
                self.zero_point.len()
            )));
        }

        let mut out = Vec::with_capacity(num_elements);

        if num_channels == 1 {
            // Per-tensor dequantization.
            let scale = self.scale[0];
            let zp = self.zero_point[0];
            for &q in data {
                out.push((q as i32 - zp) as f32 * scale);
            }
        } else {
            // Per-channel dequantization.
            let channel_size = num_elements.checked_div(num_channels).ok_or_else(|| {
                ModelError::quantization_error("num_channels is zero in per-channel dequantize")
            })?;
            if channel_size * num_channels != num_elements {
                return Err(ModelError::quantization_error(format!(
                    "data length {} is not divisible by num_channels {}",
                    num_elements, num_channels
                )));
            }
            for (ch, (&scale, &zp)) in self.scale.iter().zip(self.zero_point.iter()).enumerate() {
                let start = ch * channel_size;
                for &q in &data[start..start + channel_size] {
                    out.push((q as i32 - zp) as f32 * scale);
                }
            }
        }

        Ok(out)
    }

    fn dequantize_fp16(&self) -> ModelResult<Vec<f32>> {
        let data = self.data_f16.as_ref().ok_or_else(|| {
            ModelError::quantization_error("FP16 tensor is missing data_f16 buffer")
        })?;
        Ok(data
            .iter()
            .map(|&bits| f16::from_bits(bits).to_f32())
            .collect())
    }
}

// ---------------------------------------------------------------------------
// CalibrationData
// ---------------------------------------------------------------------------

/// Accumulates per-tensor (min, max) statistics across calibration passes.
#[derive(Debug, Default, Clone)]
pub struct CalibrationData {
    observations: HashMap<String, (f32, f32)>,
}

impl CalibrationData {
    /// Create a new, empty calibration store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Incorporate a slice of values into the running (min, max) for `name`.
    ///
    /// Calling this multiple times for the same key extends the running range.
    pub fn observe(&mut self, name: &str, values: &[f32]) {
        if values.is_empty() {
            return;
        }
        let (new_min, new_max) = values
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
                (mn.min(v), mx.max(v))
            });

        self.observations
            .entry(name.to_string())
            .and_modify(|(mn, mx)| {
                *mn = mn.min(new_min);
                *mx = mx.max(new_max);
            })
            .or_insert((new_min, new_max));
    }

    /// Return the observed (min, max) for `name`, or `None` if unseen.
    pub fn get_range(&self, name: &str) -> Option<(f32, f32)> {
        self.observations.get(name).copied()
    }
}

// ---------------------------------------------------------------------------
// ModelQuantizer
// ---------------------------------------------------------------------------

/// High-level post-training quantizer that maps weight names → [`QuantizedTensor`].
pub struct ModelQuantizer {
    config: QuantConfig,
    calibration: CalibrationData,
}

impl ModelQuantizer {
    /// Create a new quantizer with the given configuration.
    pub fn new(config: QuantConfig) -> Self {
        Self {
            config,
            calibration: CalibrationData::new(),
        }
    }

    /// Record raw values for one tensor during the calibration pass.
    ///
    /// After collecting enough samples (see [`QuantConfig::calibration_samples`])
    /// call [`Self::quantize_weights`] to apply the gathered statistics.
    pub fn calibrate_tensor(&mut self, name: &str, values: &[f32]) {
        self.calibration.observe(name, values);
    }

    /// Quantize a single flat `f32` slice to a [`QuantizedTensor`].
    ///
    /// `shape` must be consistent with `values.len()` (product of shape dims
    /// must equal `values.len()`).
    pub fn quantize_tensor(
        &self,
        name: &str,
        values: &[f32],
        shape: &[usize],
    ) -> ModelResult<QuantizedTensor> {
        // Validate shape × element count.
        let expected: usize = if shape.is_empty() {
            values.len()
        } else {
            shape.iter().product()
        };
        if expected != values.len() {
            return Err(ModelError::quantization_error(format!(
                "tensor '{}': shape {:?} implies {} elements but got {}",
                name,
                shape,
                expected,
                values.len()
            )));
        }

        match self.config.mode {
            QuantMode::Int8 => self.quantize_int8(name, values, shape),
            QuantMode::Fp16 => Self::quantize_fp16_tensor(name, values, shape),
            QuantMode::Fp32 => Err(ModelError::quantization_error(
                "Fp32 mode does not perform quantization; use the original weights directly",
            )),
        }
    }

    /// Quantize an entire weight map.
    ///
    /// Before calling this, optionally call [`Self::calibrate_tensor`] to seed
    /// calibration statistics.  Any tensor not already in calibration data will
    /// be calibrated from its own values automatically.
    pub fn quantize_weights(
        &mut self,
        weights: &HashMap<String, Vec<f32>>,
    ) -> ModelResult<HashMap<String, QuantizedTensor>> {
        // Auto-calibrate tensors that have no observations yet.
        for (name, values) in weights {
            if self.calibration.get_range(name).is_none() {
                self.calibration.observe(name, values);
            }
        }

        let mut out = HashMap::with_capacity(weights.len());
        for (name, values) in weights {
            // Treat as 1-D flat when no explicit shape is available.
            let shape = vec![values.len()];
            let qt = self.quantize_tensor(name, values, &shape)?;
            out.insert(name.clone(), qt);
        }
        Ok(out)
    }

    /// Dequantize all tensors back to flat `Vec<f32>` maps.
    ///
    /// Useful for measuring round-trip accuracy.
    pub fn dequantize_weights(
        quantized: &HashMap<String, QuantizedTensor>,
    ) -> ModelResult<HashMap<String, Vec<f32>>> {
        let mut out = HashMap::with_capacity(quantized.len());
        for (name, qt) in quantized {
            let values = qt.dequantize()?;
            out.insert(name.clone(), values);
        }
        Ok(out)
    }

    // ------------------------------------------------------------------
    // Private quantization back-ends
    // ------------------------------------------------------------------

    /// Dispatch to per-tensor or per-channel INT8 path.
    fn quantize_int8(
        &self,
        name: &str,
        values: &[f32],
        shape: &[usize],
    ) -> ModelResult<QuantizedTensor> {
        if self.config.per_channel && shape.len() >= 2 {
            self.quantize_int8_per_channel(name, values, shape)
        } else {
            self.quantize_int8_per_tensor(name, values, shape)
        }
    }

    fn quantize_int8_per_tensor(
        &self,
        name: &str,
        values: &[f32],
        shape: &[usize],
    ) -> ModelResult<QuantizedTensor> {
        // Prefer calibration statistics; fall back to computing from values.
        let (min_val, max_val) = self
            .calibration
            .get_range(name)
            .unwrap_or_else(|| compute_range(values));

        let (scale, zero_point) = if self.config.symmetric {
            symmetric_scale(min_val, max_val)
        } else {
            asymmetric_scale(min_val, max_val)
        };

        let data_i8: Vec<i8> = values
            .iter()
            .map(|&v| quantize_value_int8(v, scale, zero_point))
            .collect();

        Ok(QuantizedTensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            mode: QuantMode::Int8,
            scale: vec![scale],
            zero_point: vec![zero_point],
            data_i8: Some(data_i8),
            data_f16: None,
        })
    }

    fn quantize_int8_per_channel(
        &self,
        name: &str,
        values: &[f32],
        shape: &[usize],
    ) -> ModelResult<QuantizedTensor> {
        let num_channels = shape[0];
        if num_channels == 0 {
            return Err(ModelError::quantization_error(format!(
                "tensor '{}': per-channel quantization requires at least one channel",
                name
            )));
        }

        let channel_size = values.len() / num_channels;
        if channel_size == 0 || channel_size * num_channels != values.len() {
            return Err(ModelError::quantization_error(format!(
                "tensor '{}': {} elements not evenly divisible by {} channels",
                name,
                values.len(),
                num_channels
            )));
        }

        let mut scales = Vec::with_capacity(num_channels);
        let mut zero_points = Vec::with_capacity(num_channels);
        let mut data_i8 = Vec::with_capacity(values.len());

        for ch in 0..num_channels {
            let slice = &values[ch * channel_size..(ch + 1) * channel_size];
            let (mn, mx) = compute_range(slice);
            let (scale, zp) = if self.config.symmetric {
                symmetric_scale(mn, mx)
            } else {
                asymmetric_scale(mn, mx)
            };
            scales.push(scale);
            zero_points.push(zp);
            for &v in slice {
                data_i8.push(quantize_value_int8(v, scale, zp));
            }
        }

        Ok(QuantizedTensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            mode: QuantMode::Int8,
            scale: scales,
            zero_point: zero_points,
            data_i8: Some(data_i8),
            data_f16: None,
        })
    }

    fn quantize_fp16_tensor(
        name: &str,
        values: &[f32],
        shape: &[usize],
    ) -> ModelResult<QuantizedTensor> {
        let data_f16: Vec<u16> = values.iter().map(|&v| f32_to_f16_bits(v)).collect();

        Ok(QuantizedTensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            mode: QuantMode::Fp16,
            // FP16 does not use scale/zero_point, but fields must be non-empty.
            scale: vec![1.0_f32],
            zero_point: vec![0_i32],
            data_i8: None,
            data_f16: Some(data_f16),
        })
    }
}

// ---------------------------------------------------------------------------
// Convenience free functions
// ---------------------------------------------------------------------------

/// Quantize a weight map to INT8 using default symmetric per-tensor settings.
pub fn quantize_to_int8(
    weights: &HashMap<String, Vec<f32>>,
) -> ModelResult<HashMap<String, QuantizedTensor>> {
    let config = QuantConfig {
        mode: QuantMode::Int8,
        ..Default::default()
    };
    let mut quantizer = ModelQuantizer::new(config);
    quantizer.quantize_weights(weights)
}

/// Quantize a weight map to FP16.
pub fn quantize_to_fp16(
    weights: &HashMap<String, Vec<f32>>,
) -> ModelResult<HashMap<String, QuantizedTensor>> {
    let config = QuantConfig {
        mode: QuantMode::Fp16,
        ..Default::default()
    };
    let mut quantizer = ModelQuantizer::new(config);
    quantizer.quantize_weights(weights)
}

// ---------------------------------------------------------------------------
// Internal arithmetic helpers
// ---------------------------------------------------------------------------

/// Compute the (min, max) range of a slice; returns (0.0, 0.0) for empty slices.
fn compute_range(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0_f32, 0.0_f32);
    }
    values
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        })
}

/// Symmetric INT8 scale and zero-point (zero_point is always 0).
///
/// `scale = max(|min|, |max|) / 127`
fn symmetric_scale(min_val: f32, max_val: f32) -> (f32, i32) {
    let max_abs = min_val.abs().max(max_val.abs());
    let scale = if max_abs < f32::EPSILON {
        1.0_f32 // avoid division by zero for all-zero tensors
    } else {
        max_abs / 127.0_f32
    };
    (scale, 0_i32)
}

/// Asymmetric INT8 scale and zero-point.
///
/// Maps `[min_val, max_val]` → `[−128, 127]`.
fn asymmetric_scale(min_val: f32, max_val: f32) -> (f32, i32) {
    let range = max_val - min_val;
    if range < f32::EPSILON {
        return (1.0_f32, 0_i32);
    }
    let scale = range / 255.0_f32;
    let zero_point = (-128.0_f32 - min_val / scale)
        .round()
        .clamp(-128.0_f32, 127.0_f32) as i32;
    (scale, zero_point)
}

/// Quantize a single f32 value to INT8 given scale and zero_point.
#[inline]
fn quantize_value_int8(value: f32, scale: f32, zero_point: i32) -> i8 {
    let q = (value / scale).round() as i32 + zero_point;
    q.clamp(-127_i32, 127_i32) as i8
}

/// Convert f32 to FP16 bits using the `half` crate.
#[inline]
fn f32_to_f16_bits(v: f32) -> u16 {
    f16::from_f32(v).to_bits()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ---- QuantConfig -------------------------------------------------------

    #[test]
    fn test_quantize_config_default() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.mode, QuantMode::Int8);
        assert!(cfg.symmetric);
        assert!(!cfg.per_channel);
        assert_eq!(cfg.calibration_samples, 64);
    }

    // ---- INT8 round-trip ---------------------------------------------------

    #[test]
    fn test_int8_quantize_dequantize_round_trip() {
        let values = vec![0.0_f32, 1.0, -1.0, 0.5, -0.5];
        let config = QuantConfig {
            mode: QuantMode::Int8,
            ..Default::default()
        };
        let quantizer = ModelQuantizer::new(config);
        let qt = quantizer
            .quantize_tensor("t", &values, &[values.len()])
            .expect("quantize_tensor failed");
        let deq = qt.dequantize().expect("dequantize failed");

        assert_eq!(deq.len(), values.len());
        // Worst-case INT8 error = scale = max_abs / 127 ≈ 1/127 per step.
        let max_allowed_err = 1.0_f32 / 127.0_f32 + 1e-6_f32;
        for (orig, recovered) in values.iter().zip(deq.iter()) {
            assert!(
                (orig - recovered).abs() < max_allowed_err,
                "orig={orig} recovered={recovered}"
            );
        }
    }

    // ---- FP16 round-trip ---------------------------------------------------

    #[test]
    fn test_fp16_quantize_dequantize_round_trip() {
        let values = vec![1.0_f32, 2.0, 0.5, -3.15];
        let config = QuantConfig {
            mode: QuantMode::Fp16,
            ..Default::default()
        };
        let quantizer = ModelQuantizer::new(config);
        let qt = quantizer
            .quantize_tensor("fp16_t", &values, &[values.len()])
            .expect("quantize_tensor fp16 failed");
        let deq = qt.dequantize().expect("dequantize fp16 failed");

        assert_eq!(deq.len(), values.len());
        for (orig, recovered) in values.iter().zip(deq.iter()) {
            assert!(
                (orig - recovered).abs() < 0.01_f32,
                "orig={orig} recovered={recovered}"
            );
        }
    }

    // ---- CalibrationData ---------------------------------------------------

    #[test]
    fn test_calibration_data_observe() {
        let mut cal = CalibrationData::new();
        cal.observe("layer0.weight", &[-2.0_f32, 0.0, 3.5]);
        let (min_v, max_v) = cal
            .get_range("layer0.weight")
            .expect("range missing after observe");
        assert!((min_v - (-2.0_f32)).abs() < 1e-6_f32, "min={min_v}");
        assert!((max_v - 3.5_f32).abs() < 1e-6_f32, "max={max_v}");
    }

    #[test]
    fn test_calibration_data_running_update() {
        let mut cal = CalibrationData::new();
        cal.observe("w", &[1.0_f32, 2.0]);
        cal.observe("w", &[-5.0_f32, 4.0]);
        let (mn, mx) = cal.get_range("w").unwrap();
        assert!((mn - (-5.0_f32)).abs() < 1e-6_f32);
        assert!((mx - 4.0_f32).abs() < 1e-6_f32);
    }

    #[test]
    fn test_calibration_data_missing() {
        let cal = CalibrationData::new();
        assert!(cal.get_range("nonexistent").is_none());
    }

    // ---- quantize_weights --------------------------------------------------

    #[test]
    fn test_quantize_weights_int8() {
        let mut weights = HashMap::new();
        weights.insert("w1".to_string(), vec![1.0_f32, -1.0, 0.5, -0.5]);
        let config = QuantConfig {
            mode: QuantMode::Int8,
            ..Default::default()
        };
        let mut quantizer = ModelQuantizer::new(config);
        let quantized = quantizer
            .quantize_weights(&weights)
            .expect("quantize_weights failed");
        assert!(quantized.contains_key("w1"), "key 'w1' missing");
        assert!(
            quantized["w1"].data_i8.is_some(),
            "data_i8 should be Some for INT8"
        );
        assert!(
            quantized["w1"].data_f16.is_none(),
            "data_f16 should be None for INT8"
        );
    }

    #[test]
    fn test_quantize_weights_fp16() {
        let mut weights = HashMap::new();
        weights.insert("proj".to_string(), vec![3.15_f32, -2.71, 0.0, 1.0]);
        let config = QuantConfig {
            mode: QuantMode::Fp16,
            ..Default::default()
        };
        let mut quantizer = ModelQuantizer::new(config);
        let quantized = quantizer
            .quantize_weights(&weights)
            .expect("quantize_weights fp16 failed");
        assert!(quantized.contains_key("proj"));
        assert!(quantized["proj"].data_f16.is_some());
        assert!(quantized["proj"].data_i8.is_none());
    }

    // ---- Convenience functions ---------------------------------------------

    #[test]
    fn test_quantize_to_int8_convenience() {
        let mut weights = HashMap::new();
        weights.insert("embed".to_string(), vec![0.1_f32, 0.2, -0.1]);
        let result = quantize_to_int8(&weights).expect("quantize_to_int8 failed");
        let deq = ModelQuantizer::dequantize_weights(&result).expect("dequantize_weights failed");
        for (v_orig, v_deq) in weights["embed"].iter().zip(deq["embed"].iter()) {
            assert!(
                (v_orig - v_deq).abs() < 0.02_f32,
                "orig={v_orig} deq={v_deq}"
            );
        }
    }

    #[test]
    fn test_quantize_to_fp16_convenience() {
        let mut weights = HashMap::new();
        weights.insert("proj".to_string(), vec![3.15_f32, -2.71, 0.0, 1.0]);
        let result = quantize_to_fp16(&weights).expect("quantize_to_fp16 failed");
        let deq = ModelQuantizer::dequantize_weights(&result).expect("dequantize_weights failed");
        for (v_orig, v_deq) in weights["proj"].iter().zip(deq["proj"].iter()) {
            assert!(
                (v_orig - v_deq).abs() < 0.01_f32,
                "orig={v_orig} deq={v_deq}"
            );
        }
    }

    // ---- QuantizedTensor::dequantize directly ------------------------------

    #[test]
    fn test_quantized_tensor_dequantize() {
        let qt = QuantizedTensor {
            name: "test".to_string(),
            shape: vec![4],
            mode: QuantMode::Int8,
            scale: vec![1.0_f32 / 127.0_f32],
            zero_point: vec![0_i32],
            data_i8: Some(vec![127_i8, -127, 64, -64]),
            data_f16: None,
        };
        let values = qt.dequantize().expect("dequantize failed");
        assert_eq!(values.len(), 4);
        assert!(
            (values[0] - 1.0_f32).abs() < 0.01_f32,
            "values[0]={}",
            values[0]
        );
        assert!(
            (values[1] - (-1.0_f32)).abs() < 0.01_f32,
            "values[1]={}",
            values[1]
        );
    }

    // ---- Per-channel INT8 --------------------------------------------------

    #[test]
    fn test_per_channel_int8_round_trip() {
        // 2 channels × 4 elements each.
        let values = vec![
            10.0_f32, 20.0, 30.0, 40.0, // channel 0 (large range)
            0.1_f32, 0.2, -0.1, -0.2, // channel 1 (small range)
        ];
        let shape = vec![2, 4];
        let config = QuantConfig {
            mode: QuantMode::Int8,
            per_channel: true,
            symmetric: true,
            ..Default::default()
        };
        let quantizer = ModelQuantizer::new(config);
        let qt = quantizer
            .quantize_tensor("pc_test", &values, &shape)
            .expect("per-channel quantize failed");

        assert_eq!(qt.scale.len(), 2, "should have 2 channel scales");
        let deq = qt.dequantize().expect("dequantize failed");
        assert_eq!(deq.len(), values.len());
        for (orig, recovered) in values.iter().zip(deq.iter()) {
            assert!(
                (orig - recovered).abs() < 1.0_f32,
                "orig={orig} recovered={recovered}"
            );
        }
    }

    // ---- Calibrate then quantize -------------------------------------------

    #[test]
    fn test_calibrate_then_quantize() {
        let mut quantizer = ModelQuantizer::new(QuantConfig::default());
        // Observe a wider range during calibration than in the actual weights.
        quantizer.calibrate_tensor("layer", &[-100.0_f32, 100.0]);
        let values = vec![1.0_f32, -1.0, 0.5];
        let qt = quantizer
            .quantize_tensor("layer", &values, &[3])
            .expect("calibrated quantize failed");
        // Scale should reflect the calibrated range, not just the narrow values.
        assert!(
            qt.scale[0] > 1.0_f32 / 127.0_f32,
            "scale should reflect calibrated range"
        );
    }

    // ---- Edge cases --------------------------------------------------------

    #[test]
    fn test_all_zeros_int8() {
        let values = vec![0.0_f32; 8];
        let mut weights = HashMap::new();
        weights.insert("zeros".to_string(), values);
        let qt = quantize_to_int8(&weights).expect("quantize zeros failed");
        let deq = ModelQuantizer::dequantize_weights(&qt).expect("dequantize zeros failed");
        for v in &deq["zeros"] {
            assert_eq!(*v, 0.0_f32, "all-zero tensor should dequantize to zero");
        }
    }
}
