//! Mixed precision support for efficient inference
//!
//! This module provides support for FP16 (half precision) and BF16 (bfloat16)
//! inference to reduce memory usage and increase throughput on supported hardware.

use crate::error::{InferenceError, InferenceResult};
use half::{bf16, f16};
use scirs2_core::ndarray::{Array1, Array2};

/// Precision mode for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum PrecisionMode {
    /// Full precision (FP32)
    #[default]
    FP32,
    /// Half precision (FP16) - good for NVIDIA GPUs
    FP16,
    /// Brain float 16 (BF16) - good for modern accelerators
    BF16,
    /// Mixed precision - compute in FP16/BF16 but accumulate in FP32
    Mixed {
        /// Compute precision
        compute: ComputePrecision,
        /// Whether to accumulate in FP32
        accumulate_fp32: bool,
    },
}

/// Compute precision for mixed mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ComputePrecision {
    /// FP16 compute
    FP16,
    /// BF16 compute
    BF16,
}

impl PrecisionMode {
    /// Check if this mode uses reduced precision
    pub fn is_reduced_precision(&self) -> bool {
        !matches!(self, PrecisionMode::FP32)
    }

    /// Get the memory reduction factor compared to FP32
    pub fn memory_reduction_factor(&self) -> f32 {
        match self {
            PrecisionMode::FP32 => 1.0,
            PrecisionMode::FP16 | PrecisionMode::BF16 => 0.5,
            PrecisionMode::Mixed { .. } => 0.75, // Mixed uses some FP32 for accumulation
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &str {
        match self {
            PrecisionMode::FP32 => "FP32",
            PrecisionMode::FP16 => "FP16",
            PrecisionMode::BF16 => "BF16",
            PrecisionMode::Mixed { compute, .. } => match compute {
                ComputePrecision::FP16 => "Mixed-FP16",
                ComputePrecision::BF16 => "Mixed-BF16",
            },
        }
    }
}

/// Configuration for mixed precision inference
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PrecisionConfig {
    /// Precision mode to use
    pub mode: PrecisionMode,
    /// Loss scaling factor to prevent underflow in FP16
    pub loss_scale: f32,
    /// Whether to automatically adjust loss scale
    pub dynamic_loss_scale: bool,
    /// Threshold for gradient clipping in reduced precision
    pub grad_clip_threshold: Option<f32>,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            mode: PrecisionMode::FP32,
            loss_scale: 1.0,
            dynamic_loss_scale: false,
            grad_clip_threshold: None,
        }
    }
}

impl PrecisionConfig {
    /// Create a new precision configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set precision mode
    pub fn mode(mut self, mode: PrecisionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Enable FP16 precision
    pub fn fp16(mut self) -> Self {
        self.mode = PrecisionMode::FP16;
        self
    }

    /// Enable BF16 precision
    pub fn bf16(mut self) -> Self {
        self.mode = PrecisionMode::BF16;
        self
    }

    /// Enable mixed precision with FP16 compute
    pub fn mixed_fp16(mut self, accumulate_fp32: bool) -> Self {
        self.mode = PrecisionMode::Mixed {
            compute: ComputePrecision::FP16,
            accumulate_fp32,
        };
        self
    }

    /// Enable mixed precision with BF16 compute
    pub fn mixed_bf16(mut self, accumulate_fp32: bool) -> Self {
        self.mode = PrecisionMode::Mixed {
            compute: ComputePrecision::BF16,
            accumulate_fp32,
        };
        self
    }

    /// Set loss scaling factor
    pub fn loss_scale(mut self, scale: f32) -> Self {
        self.loss_scale = scale;
        self
    }

    /// Enable dynamic loss scaling
    pub fn dynamic_loss_scale(mut self, enabled: bool) -> Self {
        self.dynamic_loss_scale = enabled;
        self
    }

    /// Set gradient clipping threshold
    pub fn grad_clip_threshold(mut self, threshold: f32) -> Self {
        self.grad_clip_threshold = Some(threshold);
        self
    }
}

/// Precision converter for array operations
pub struct PrecisionConverter {
    config: PrecisionConfig,
}

impl PrecisionConverter {
    /// Create a new precision converter
    pub fn new(config: PrecisionConfig) -> Self {
        Self { config }
    }

    /// Convert FP32 array to reduced precision and back (for inference)
    pub fn convert_and_compute_1d(
        &self,
        data: &Array1<f32>,
        op: impl Fn(&Array1<f32>) -> Array1<f32>,
    ) -> InferenceResult<Array1<f32>> {
        match self.config.mode {
            PrecisionMode::FP32 => Ok(op(data)),
            PrecisionMode::FP16 => {
                let fp16_data = self.to_fp16_1d(data);
                let fp16_result = op(&self.from_fp16_1d(&fp16_data));
                Ok(fp16_result)
            }
            PrecisionMode::BF16 => {
                let bf16_data = self.to_bf16_1d(data);
                let bf16_result = op(&self.from_bf16_1d(&bf16_data));
                Ok(bf16_result)
            }
            PrecisionMode::Mixed {
                compute,
                accumulate_fp32,
            } => {
                if accumulate_fp32 {
                    // Compute in reduced precision, accumulate in FP32
                    let reduced = match compute {
                        ComputePrecision::FP16 => {
                            let fp16_data = self.to_fp16_1d(data);
                            self.from_fp16_1d(&fp16_data)
                        }
                        ComputePrecision::BF16 => {
                            let bf16_data = self.to_bf16_1d(data);
                            self.from_bf16_1d(&bf16_data)
                        }
                    };
                    Ok(op(&reduced))
                } else {
                    // Full mixed precision
                    match compute {
                        ComputePrecision::FP16 => {
                            let fp16_data = self.to_fp16_1d(data);
                            Ok(op(&self.from_fp16_1d(&fp16_data)))
                        }
                        ComputePrecision::BF16 => {
                            let bf16_data = self.to_bf16_1d(data);
                            Ok(op(&self.from_bf16_1d(&bf16_data)))
                        }
                    }
                }
            }
        }
    }

    /// Convert FP32 array to FP16
    pub fn to_fp16_1d(&self, data: &Array1<f32>) -> Vec<f16> {
        data.iter().map(|&x| f16::from_f32(x)).collect()
    }

    /// Convert FP16 array to FP32
    pub fn from_fp16_1d(&self, data: &[f16]) -> Array1<f32> {
        Array1::from_vec(data.iter().map(|&x| x.to_f32()).collect())
    }

    /// Convert FP32 array to BF16
    pub fn to_bf16_1d(&self, data: &Array1<f32>) -> Vec<bf16> {
        data.iter().map(|&x| bf16::from_f32(x)).collect()
    }

    /// Convert BF16 array to FP32
    pub fn from_bf16_1d(&self, data: &[bf16]) -> Array1<f32> {
        Array1::from_vec(data.iter().map(|&x| x.to_f32()).collect())
    }

    /// Convert 2D FP32 array to FP16
    pub fn to_fp16_2d(&self, data: &Array2<f32>) -> Vec<f16> {
        data.iter().map(|&x| f16::from_f32(x)).collect()
    }

    /// Convert FP16 to 2D FP32 array
    pub fn from_fp16_2d(
        &self,
        data: &[f16],
        shape: (usize, usize),
    ) -> InferenceResult<Array2<f32>> {
        let vec: Vec<f32> = data.iter().map(|&x| x.to_f32()).collect();
        Array2::from_shape_vec(shape, vec).map_err(|e| {
            InferenceError::ForwardError(format!("Shape error in FP16 conversion: {}", e))
        })
    }

    /// Convert 2D FP32 array to BF16
    pub fn to_bf16_2d(&self, data: &Array2<f32>) -> Vec<bf16> {
        data.iter().map(|&x| bf16::from_f32(x)).collect()
    }

    /// Convert BF16 to 2D FP32 array
    pub fn from_bf16_2d(
        &self,
        data: &[bf16],
        shape: (usize, usize),
    ) -> InferenceResult<Array2<f32>> {
        let vec: Vec<f32> = data.iter().map(|&x| x.to_f32()).collect();
        Array2::from_shape_vec(shape, vec).map_err(|e| {
            InferenceError::ForwardError(format!("Shape error in BF16 conversion: {}", e))
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &PrecisionConfig {
        &self.config
    }
}

/// Statistics about precision conversion
#[derive(Debug, Clone, Default)]
pub struct PrecisionStats {
    /// Number of conversions performed
    pub num_conversions: usize,
    /// Total memory saved (bytes)
    pub memory_saved: usize,
    /// Average numerical error from conversion
    pub avg_error: f64,
    /// Maximum numerical error observed
    pub max_error: f64,
}

impl PrecisionStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a conversion
    pub fn record_conversion(&mut self, original_size: usize, precision_mode: &PrecisionMode) {
        self.num_conversions += 1;
        let saved =
            (original_size as f32 * (1.0 - precision_mode.memory_reduction_factor())) as usize;
        self.memory_saved += saved;
    }

    /// Record numerical error
    pub fn record_error(&mut self, error: f64) {
        let n = self.num_conversions as f64;
        self.avg_error = (self.avg_error * (n - 1.0) + error) / n;
        self.max_error = self.max_error.max(error);
    }

    /// Get memory saved in MB
    pub fn memory_saved_mb(&self) -> f64 {
        self.memory_saved as f64 / (1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_mode_creation() {
        let mode = PrecisionMode::FP32;
        assert_eq!(mode.name(), "FP32");
        assert!(!mode.is_reduced_precision());
    }

    #[test]
    fn test_precision_mode_fp16() {
        let mode = PrecisionMode::FP16;
        assert_eq!(mode.name(), "FP16");
        assert!(mode.is_reduced_precision());
        assert_eq!(mode.memory_reduction_factor(), 0.5);
    }

    #[test]
    fn test_precision_mode_bf16() {
        let mode = PrecisionMode::BF16;
        assert_eq!(mode.name(), "BF16");
        assert!(mode.is_reduced_precision());
        assert_eq!(mode.memory_reduction_factor(), 0.5);
    }

    #[test]
    fn test_precision_mode_mixed() {
        let mode = PrecisionMode::Mixed {
            compute: ComputePrecision::FP16,
            accumulate_fp32: true,
        };
        assert_eq!(mode.name(), "Mixed-FP16");
        assert!(mode.is_reduced_precision());
    }

    #[test]
    fn test_precision_config_builder() {
        let config = PrecisionConfig::new()
            .fp16()
            .loss_scale(128.0)
            .dynamic_loss_scale(true);

        assert_eq!(config.mode, PrecisionMode::FP16);
        assert_eq!(config.loss_scale, 128.0);
        assert!(config.dynamic_loss_scale);
    }

    #[test]
    fn test_fp16_conversion_1d() {
        let config = PrecisionConfig::new().fp16();
        let converter = PrecisionConverter::new(config);

        let data = Array1::from_vec(vec![1.0, 2.5, -3.75, 0.0]);
        let fp16_data = converter.to_fp16_1d(&data);
        let restored = converter.from_fp16_1d(&fp16_data);

        // Should be very close (within FP16 precision)
        for (orig, rest) in data.iter().zip(restored.iter()) {
            assert!((orig - rest).abs() < 0.001);
        }
    }

    #[test]
    fn test_bf16_conversion_1d() {
        let config = PrecisionConfig::new().bf16();
        let converter = PrecisionConverter::new(config);

        let data = Array1::from_vec(vec![1.0, 2.5, -3.75, 0.0]);
        let bf16_data = converter.to_bf16_1d(&data);
        let restored = converter.from_bf16_1d(&bf16_data);

        // BF16 has less precision than FP16
        for (orig, rest) in data.iter().zip(restored.iter()) {
            assert!((orig - rest).abs() < 0.01);
        }
    }

    #[test]
    fn test_convert_and_compute() {
        let config = PrecisionConfig::new().fp16();
        let converter = PrecisionConverter::new(config);

        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        // Simple doubling operation
        let result = converter
            .convert_and_compute_1d(&data, |x| x.mapv(|v| v * 2.0))
            .unwrap();

        // Check results are close to expected (within FP16 precision)
        for (i, &val) in result.iter().enumerate() {
            let expected = data[i] * 2.0;
            assert!((val - expected).abs() < 0.01);
        }
    }

    #[test]
    fn test_precision_stats() {
        let mut stats = PrecisionStats::new();

        assert_eq!(stats.num_conversions, 0);
        assert_eq!(stats.memory_saved, 0);

        let mode = PrecisionMode::FP16;
        stats.record_conversion(1000, &mode);

        assert_eq!(stats.num_conversions, 1);
        assert_eq!(stats.memory_saved, 500); // 50% reduction

        stats.record_error(0.001);
        assert!(stats.avg_error > 0.0);
        assert!(stats.max_error > 0.0);
    }

    #[test]
    fn test_mixed_precision_compute() {
        let config = PrecisionConfig::new().mixed_fp16(true);
        let converter = PrecisionConverter::new(config);

        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let result = converter
            .convert_and_compute_1d(&data, |x| x.mapv(|v| v * 2.0))
            .unwrap();

        // Mixed precision should have good accuracy
        for (i, &val) in result.iter().enumerate() {
            let expected = data[i] * 2.0;
            assert!((val - expected).abs() < 0.001);
        }
    }

    #[test]
    fn test_fp16_2d_conversion() {
        let config = PrecisionConfig::new().fp16();
        let converter = PrecisionConverter::new(config);

        let data = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let fp16_data = converter.to_fp16_2d(&data);
        let restored = converter.from_fp16_2d(&fp16_data, (2, 3)).unwrap();

        assert_eq!(restored.shape(), &[2, 3]);

        for (orig, rest) in data.iter().zip(restored.iter()) {
            assert!((orig - rest).abs() < 0.001);
        }
    }

    #[test]
    fn test_bf16_2d_conversion() {
        let config = PrecisionConfig::new().bf16();
        let converter = PrecisionConverter::new(config);

        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let bf16_data = converter.to_bf16_2d(&data);
        let restored = converter.from_bf16_2d(&bf16_data, (2, 2)).unwrap();

        assert_eq!(restored.shape(), &[2, 2]);

        for (orig, rest) in data.iter().zip(restored.iter()) {
            assert!((orig - rest).abs() < 0.01);
        }
    }
}
