//! Sensor calibration and data correction utilities
//!
//! This module provides tools for calibrating sensors and correcting systematic
//! errors in measurement data.

use crate::error::{IoError, IoResult};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Calibration curve for non-linear sensor response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationCurve {
    /// Input points (raw sensor values)
    pub input_points: Vec<f32>,
    /// Output points (calibrated values)
    pub output_points: Vec<f32>,
}

impl CalibrationCurve {
    /// Create new calibration curve
    pub fn new(input_points: Vec<f32>, output_points: Vec<f32>) -> IoResult<Self> {
        if input_points.len() != output_points.len() {
            return Err(IoError::InvalidConfig(
                "Input and output points must have same length".to_string(),
            ));
        }
        if input_points.len() < 2 {
            return Err(IoError::InvalidConfig(
                "Need at least 2 calibration points".to_string(),
            ));
        }

        Ok(Self {
            input_points,
            output_points,
        })
    }

    /// Apply calibration using linear interpolation
    pub fn apply(&self, value: f32) -> f32 {
        // Find bracketing points
        let mut lower_idx = 0;
        let mut upper_idx = self.input_points.len() - 1;

        for (i, &input) in self.input_points.iter().enumerate() {
            if value >= input {
                lower_idx = i;
            }
            if value <= input && i < upper_idx {
                upper_idx = i;
                break;
            }
        }

        if lower_idx == upper_idx {
            return self.output_points[lower_idx];
        }

        // Linear interpolation
        let x0 = self.input_points[lower_idx];
        let x1 = self.input_points[upper_idx];
        let y0 = self.output_points[lower_idx];
        let y1 = self.output_points[upper_idx];

        if (x1 - x0).abs() < 1e-10 {
            return y0;
        }

        let t = (value - x0) / (x1 - x0);
        y0 + t * (y1 - y0)
    }
}

/// Calibration parameters for a sensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationParams {
    /// Zero offset (bias)
    pub offset: f32,
    /// Scale factor (gain)
    pub scale: f32,
    /// Temperature coefficient (optional)
    pub temp_coefficient: Option<f32>,
    /// Non-linear calibration curve (optional)
    pub curve: Option<CalibrationCurve>,
    /// Custom parameters
    pub custom: HashMap<String, f32>,
}

impl Default for CalibrationParams {
    fn default() -> Self {
        Self {
            offset: 0.0,
            scale: 1.0,
            temp_coefficient: None,
            curve: None,
            custom: HashMap::new(),
        }
    }
}

impl CalibrationParams {
    /// Create identity calibration (no correction)
    pub fn identity() -> Self {
        Self::default()
    }

    /// Create calibration from offset and scale
    pub fn from_offset_scale(offset: f32, scale: f32) -> Self {
        Self {
            offset,
            scale,
            ..Default::default()
        }
    }

    /// Apply calibration to single value
    pub fn calibrate(&self, raw_value: f32, temperature: Option<f32>) -> f32 {
        let mut value = raw_value;

        // Apply offset and scale
        value = (value - self.offset) * self.scale;

        // Apply temperature compensation
        if let (Some(temp_coef), Some(temp)) = (self.temp_coefficient, temperature) {
            value *= 1.0 + temp_coef * (temp - 25.0); // 25°C reference
        }

        // Apply non-linear curve
        if let Some(ref curve) = self.curve {
            value = curve.apply(value);
        }

        value
    }

    /// Apply calibration to array of values
    pub fn calibrate_array(&self, raw_values: &[f32], temperature: Option<f32>) -> Array1<f32> {
        Array1::from_vec(
            raw_values
                .iter()
                .map(|&v| self.calibrate(v, temperature))
                .collect(),
        )
    }
}

/// Multi-point calibration using least squares
pub struct MultiPointCalibrator {
    /// Reference values
    reference: Vec<f32>,
    /// Measured values
    measured: Vec<f32>,
}

impl MultiPointCalibrator {
    /// Create new multi-point calibrator
    pub fn new() -> Self {
        Self {
            reference: Vec::new(),
            measured: Vec::new(),
        }
    }

    /// Add calibration point
    pub fn add_point(&mut self, reference: f32, measured: f32) {
        self.reference.push(reference);
        self.measured.push(measured);
    }

    /// Compute calibration parameters using least squares
    pub fn compute_calibration(&self) -> IoResult<CalibrationParams> {
        if self.reference.len() < 2 {
            return Err(IoError::InvalidConfig(
                "Need at least 2 calibration points".to_string(),
            ));
        }

        // Compute mean
        let ref_mean: f32 = self.reference.iter().sum::<f32>() / self.reference.len() as f32;
        let meas_mean: f32 = self.measured.iter().sum::<f32>() / self.measured.len() as f32;

        // Compute covariance and variance
        let mut covariance = 0.0f32;
        let mut variance = 0.0f32;

        for (&r, &m) in self.reference.iter().zip(self.measured.iter()) {
            let r_diff = r - ref_mean;
            let m_diff = m - meas_mean;
            covariance += r_diff * m_diff;
            variance += m_diff * m_diff;
        }

        if variance.abs() < 1e-10 {
            return Err(IoError::InvalidConfig(
                "Measured values have zero variance".to_string(),
            ));
        }

        // Compute slope and intercept
        // Linear regression: reference = a * measured + b
        // where a = Cov(ref, meas) / Var(meas), b = ref_mean - a * meas_mean
        // Calibration formula: calibrated = (raw - offset) * scale = reference
        // Matching: scale = a, offset = meas_mean - ref_mean / scale
        let scale = covariance / variance;
        let offset = meas_mean - ref_mean / scale;

        Ok(CalibrationParams::from_offset_scale(offset, scale))
    }

    /// Get number of calibration points
    pub fn num_points(&self) -> usize {
        self.reference.len()
    }

    /// Clear all calibration points
    pub fn clear(&mut self) {
        self.reference.clear();
        self.measured.clear();
    }
}

impl Default for MultiPointCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Calibration manager for multiple sensors
pub struct CalibrationManager {
    calibrations: HashMap<String, CalibrationParams>,
}

impl CalibrationManager {
    /// Create new calibration manager
    pub fn new() -> Self {
        Self {
            calibrations: HashMap::new(),
        }
    }

    /// Add calibration for a sensor
    pub fn add_calibration(&mut self, sensor_id: String, params: CalibrationParams) {
        self.calibrations.insert(sensor_id, params);
    }

    /// Get calibration for a sensor
    pub fn get_calibration(&self, sensor_id: &str) -> Option<&CalibrationParams> {
        self.calibrations.get(sensor_id)
    }

    /// Remove calibration for a sensor
    pub fn remove_calibration(&mut self, sensor_id: &str) -> Option<CalibrationParams> {
        self.calibrations.remove(sensor_id)
    }

    /// Calibrate value from a sensor
    pub fn calibrate(&self, sensor_id: &str, raw_value: f32, temperature: Option<f32>) -> f32 {
        self.calibrations
            .get(sensor_id)
            .map(|cal| cal.calibrate(raw_value, temperature))
            .unwrap_or(raw_value) // Return raw if no calibration
    }

    /// Calibrate array from a sensor
    pub fn calibrate_array(
        &self,
        sensor_id: &str,
        raw_values: &[f32],
        temperature: Option<f32>,
    ) -> Array1<f32> {
        self.calibrations
            .get(sensor_id)
            .map(|cal| cal.calibrate_array(raw_values, temperature))
            .unwrap_or_else(|| Array1::from_vec(raw_values.to_vec()))
    }

    /// Get number of registered sensors
    pub fn num_sensors(&self) -> usize {
        self.calibrations.len()
    }

    /// Save calibrations to JSON
    pub fn to_json(&self) -> IoResult<String> {
        serde_json::to_string_pretty(&self.calibrations).map_err(IoError::JsonError)
    }

    /// Load calibrations from JSON
    pub fn from_json(json: &str) -> IoResult<Self> {
        let calibrations: HashMap<String, CalibrationParams> =
            serde_json::from_str(json).map_err(IoError::JsonError)?;
        Ok(Self { calibrations })
    }
}

impl Default for CalibrationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Auto-calibration using known reference signal
pub struct AutoCalibrator {
    /// Expected reference signal
    reference: Vec<f32>,
    /// Collected samples for calibration
    samples: Vec<Vec<f32>>,
    /// Number of samples to collect
    num_samples: usize,
}

impl AutoCalibrator {
    /// Create new auto-calibrator
    pub fn new(reference: Vec<f32>, num_samples: usize) -> Self {
        Self {
            reference,
            samples: Vec::new(),
            num_samples,
        }
    }

    /// Add measured sample
    pub fn add_sample(&mut self, sample: Vec<f32>) -> IoResult<()> {
        if sample.len() != self.reference.len() {
            return Err(IoError::InvalidConfig("Sample length mismatch".to_string()));
        }
        self.samples.push(sample);
        Ok(())
    }

    /// Check if ready to calibrate
    pub fn is_ready(&self) -> bool {
        self.samples.len() >= self.num_samples
    }

    /// Compute calibration from collected samples
    pub fn compute_calibration(&self) -> IoResult<CalibrationParams> {
        if !self.is_ready() {
            return Err(IoError::InvalidConfig(
                "Not enough samples collected".to_string(),
            ));
        }

        // Average all samples
        let mut averaged = vec![0.0f32; self.reference.len()];
        for sample in &self.samples {
            for (avg, &val) in averaged.iter_mut().zip(sample.iter()) {
                *avg += val;
            }
        }
        for avg in &mut averaged {
            *avg /= self.samples.len() as f32;
        }

        // Use multi-point calibration
        let mut calibrator = MultiPointCalibrator::new();
        for (&ref_val, &meas_val) in self.reference.iter().zip(averaged.iter()) {
            calibrator.add_point(ref_val, meas_val);
        }

        calibrator.compute_calibration()
    }

    /// Get number of samples collected
    pub fn num_collected(&self) -> usize {
        self.samples.len()
    }

    /// Reset calibrator
    pub fn reset(&mut self) {
        self.samples.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_params_basic() {
        let params = CalibrationParams::from_offset_scale(10.0, 2.0);

        // Test calibration: (raw - offset) * scale
        // (30 - 10) * 2.0 = 40.0
        let calibrated = params.calibrate(30.0, None);
        assert_eq!(calibrated, 40.0);
    }

    #[test]
    fn test_calibration_curve() {
        let curve = CalibrationCurve::new(vec![0.0, 1.0, 2.0], vec![0.0, 10.0, 30.0]).unwrap();

        // Test interpolation
        assert_eq!(curve.apply(0.0), 0.0);
        assert_eq!(curve.apply(1.0), 10.0);
        assert_eq!(curve.apply(0.5), 5.0);
        assert_eq!(curve.apply(1.5), 20.0);
    }

    #[test]
    fn test_multipoint_calibrator() {
        let mut calibrator = MultiPointCalibrator::new();

        // Add points where reference = measured * 2
        // So measured values need calibration of scale=2, offset=0
        calibrator.add_point(2.0, 1.0); // ref=2, meas=1
        calibrator.add_point(4.0, 2.0); // ref=4, meas=2
        calibrator.add_point(6.0, 3.0); // ref=6, meas=3

        let params = calibrator.compute_calibration().unwrap();

        // Verify calibration: (1.0 - offset) * scale ≈ 2.0
        let calibrated = params.calibrate(1.0, None);
        assert!((calibrated - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_calibration_manager() {
        let mut manager = CalibrationManager::new();

        let params1 = CalibrationParams::from_offset_scale(0.0, 2.0);
        let params2 = CalibrationParams::from_offset_scale(10.0, 1.0);

        manager.add_calibration("sensor1".to_string(), params1);
        manager.add_calibration("sensor2".to_string(), params2);

        assert_eq!(manager.num_sensors(), 2);

        let cal1 = manager.calibrate("sensor1", 5.0, None);
        assert_eq!(cal1, 10.0); // 5.0 * 2.0

        let cal2 = manager.calibrate("sensor2", 20.0, None);
        assert_eq!(cal2, 10.0); // (20.0 - 10.0) * 1.0
    }

    #[test]
    fn test_calibration_json_roundtrip() {
        let mut manager = CalibrationManager::new();
        manager.add_calibration(
            "test_sensor".to_string(),
            CalibrationParams::from_offset_scale(1.0, 2.0),
        );

        let json = manager.to_json().unwrap();
        let loaded = CalibrationManager::from_json(&json).unwrap();

        assert_eq!(loaded.num_sensors(), 1);
        let cal = loaded.calibrate("test_sensor", 5.0, None);
        assert_eq!(cal, 8.0); // (5.0 - 1.0) * 2.0
    }

    #[test]
    fn test_auto_calibrator() {
        let reference = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut calibrator = AutoCalibrator::new(reference.clone(), 3);

        assert!(!calibrator.is_ready());

        // Add samples with known relationship: measured = reference * 2
        // So we need scale=0.5 to recover reference from measured
        calibrator
            .add_sample(vec![2.0, 4.0, 6.0, 8.0, 10.0])
            .unwrap();
        calibrator
            .add_sample(vec![2.0, 4.0, 6.0, 8.0, 10.0])
            .unwrap();
        calibrator
            .add_sample(vec![2.0, 4.0, 6.0, 8.0, 10.0])
            .unwrap();

        assert!(calibrator.is_ready());

        let params = calibrator.compute_calibration().unwrap();

        // Calibrate measured value 2.0 should give reference 1.0
        let calibrated = params.calibrate(2.0, None);
        assert!((calibrated - 1.0).abs() < 0.2);
    }
}
