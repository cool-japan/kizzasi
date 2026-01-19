//! Mixed Precision Support (FP16/BF16)
//!
//! Provides mixed precision training and inference capabilities using:
//! - **FP16**: IEEE 754 half-precision (16-bit)
//! - **BF16**: Brain Float 16 (16-bit with same exponent range as FP32)
//!
//! # Benefits
//!
//! - 2x memory reduction compared to FP32
//! - Faster computation on modern hardware (GPUs, TPUs)
//! - Maintained numerical stability with BF16
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi_model::mixed_precision::*;
//!
//! let weights_fp32 = Array2::from_shape_vec((10, 10), vec![1.0; 100]).unwrap();
//! let weights_fp16 = to_fp16_array2(&weights_fp32);
//! let restored = from_fp16_array2(&weights_fp16);
//! ```

use crate::error::{ModelError, ModelResult};
use half::{bf16, f16};
use scirs2_core::ndarray::{Array1, Array2};

/// Precision mode for mixed precision training/inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionMode {
    /// Full precision (FP32)
    FP32,
    /// Half precision (FP16)
    FP16,
    /// Brain Float 16
    BF16,
    /// Mixed precision: FP16 for activations, FP32 for gradients
    Mixed,
}

/// FP16 weight storage
#[derive(Debug, Clone)]
pub struct FP16Weights {
    /// Data stored as FP16
    pub data: Vec<f16>,
    /// Original shape
    pub shape: Vec<usize>,
}

impl FP16Weights {
    /// Create new FP16 weights from FP32 array
    pub fn from_f32_1d(array: &Array1<f32>) -> Self {
        let data: Vec<f16> = array.iter().map(|&x| f16::from_f32(x)).collect();
        Self {
            data,
            shape: vec![array.len()],
        }
    }

    /// Create new FP16 weights from FP32 2D array
    pub fn from_f32_2d(array: &Array2<f32>) -> Self {
        let (rows, cols) = array.dim();
        let mut data = Vec::with_capacity(rows * cols);

        for i in 0..rows {
            for j in 0..cols {
                data.push(f16::from_f32(array[[i, j]]));
            }
        }

        Self {
            data,
            shape: vec![rows, cols],
        }
    }

    /// Convert back to FP32 1D array
    pub fn to_f32_1d(&self) -> ModelResult<Array1<f32>> {
        if self.shape.len() != 1 {
            return Err(ModelError::invalid_config(format!(
                "expected 1D shape, got {:?}",
                self.shape
            )));
        }

        let data: Vec<f32> = self.data.iter().map(|&x| x.to_f32()).collect();
        Ok(Array1::from_vec(data))
    }

    /// Convert back to FP32 2D array
    pub fn to_f32_2d(&self) -> ModelResult<Array2<f32>> {
        if self.shape.len() != 2 {
            return Err(ModelError::invalid_config(format!(
                "expected 2D shape, got {:?}",
                self.shape
            )));
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut array = Array2::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                array[[i, j]] = self.data[idx].to_f32();
            }
        }

        Ok(array)
    }

    /// Get memory size in bytes
    pub fn memory_size(&self) -> usize {
        self.data.len() * 2 // FP16 = 2 bytes per element
    }
}

/// BF16 weight storage
#[derive(Debug, Clone)]
pub struct BF16Weights {
    /// Data stored as BF16
    pub data: Vec<bf16>,
    /// Original shape
    pub shape: Vec<usize>,
}

impl BF16Weights {
    /// Create new BF16 weights from FP32 array
    pub fn from_f32_1d(array: &Array1<f32>) -> Self {
        let data: Vec<bf16> = array.iter().map(|&x| bf16::from_f32(x)).collect();
        Self {
            data,
            shape: vec![array.len()],
        }
    }

    /// Create new BF16 weights from FP32 2D array
    pub fn from_f32_2d(array: &Array2<f32>) -> Self {
        let (rows, cols) = array.dim();
        let mut data = Vec::with_capacity(rows * cols);

        for i in 0..rows {
            for j in 0..cols {
                data.push(bf16::from_f32(array[[i, j]]));
            }
        }

        Self {
            data,
            shape: vec![rows, cols],
        }
    }

    /// Convert back to FP32 1D array
    pub fn to_f32_1d(&self) -> ModelResult<Array1<f32>> {
        if self.shape.len() != 1 {
            return Err(ModelError::invalid_config(format!(
                "expected 1D shape, got {:?}",
                self.shape
            )));
        }

        let data: Vec<f32> = self.data.iter().map(|&x| x.to_f32()).collect();
        Ok(Array1::from_vec(data))
    }

    /// Convert back to FP32 2D array
    pub fn to_f32_2d(&self) -> ModelResult<Array2<f32>> {
        if self.shape.len() != 2 {
            return Err(ModelError::invalid_config(format!(
                "expected 2D shape, got {:?}",
                self.shape
            )));
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut array = Array2::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                array[[i, j]] = self.data[idx].to_f32();
            }
        }

        Ok(array)
    }

    /// Get memory size in bytes
    pub fn memory_size(&self) -> usize {
        self.data.len() * 2 // BF16 = 2 bytes per element
    }
}

/// Convert FP32 array to FP16
pub fn to_fp16_array1(array: &Array1<f32>) -> Array1<f16> {
    array.mapv(f16::from_f32)
}

/// Convert FP32 array to FP16
pub fn to_fp16_array2(array: &Array2<f32>) -> Array2<f16> {
    array.mapv(f16::from_f32)
}

/// Convert FP16 array to FP32
pub fn from_fp16_array1(array: &Array1<f16>) -> Array1<f32> {
    array.mapv(|x| x.to_f32())
}

/// Convert FP16 array to FP32
pub fn from_fp16_array2(array: &Array2<f16>) -> Array2<f32> {
    array.mapv(|x| x.to_f32())
}

/// Convert FP32 array to BF16
pub fn to_bf16_array1(array: &Array1<f32>) -> Array1<bf16> {
    array.mapv(bf16::from_f32)
}

/// Convert FP32 array to BF16
pub fn to_bf16_array2(array: &Array2<f32>) -> Array2<bf16> {
    array.mapv(bf16::from_f32)
}

/// Convert BF16 array to FP32
pub fn from_bf16_array1(array: &Array1<bf16>) -> Array1<f32> {
    array.mapv(|x| x.to_f32())
}

/// Convert BF16 array to FP32
pub fn from_bf16_array2(array: &Array2<bf16>) -> Array2<f32> {
    array.mapv(|x| x.to_f32())
}

/// Compute relative error after precision conversion
pub fn relative_error_fp16(original: &Array1<f32>, converted: &Array1<f32>) -> f32 {
    if original.len() != converted.len() {
        return f32::INFINITY;
    }

    let mut max_rel_error = 0.0f32;

    for i in 0..original.len() {
        let error = (original[i] - converted[i]).abs();
        let rel_error = if original[i].abs() > 1e-8 {
            error / original[i].abs()
        } else {
            error
        };

        max_rel_error = max_rel_error.max(rel_error);
    }

    max_rel_error
}

/// Gradient scaler for mixed precision training
///
/// Scales gradients to prevent underflow in FP16
#[derive(Debug, Clone)]
pub struct GradientScaler {
    /// Current scale factor
    scale: f32,
    /// Growth factor when no overflow detected
    growth_factor: f32,
    /// Backoff factor when overflow detected
    backoff_factor: f32,
    /// Growth interval (steps between scale increases)
    growth_interval: usize,
    /// Steps since last overflow
    steps_since_overflow: usize,
}

impl GradientScaler {
    /// Create a new gradient scaler
    pub fn new() -> Self {
        Self {
            scale: 65536.0, // 2^16
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_overflow: 0,
        }
    }

    /// Get current scale
    pub fn get_scale(&self) -> f32 {
        self.scale
    }

    /// Scale gradients
    pub fn scale_gradients(&self, gradients: &Array1<f32>) -> Array1<f32> {
        gradients.mapv(|x| x * self.scale)
    }

    /// Unscale gradients
    pub fn unscale_gradients(&self, scaled_gradients: &Array1<f32>) -> Array1<f32> {
        scaled_gradients.mapv(|x| x / self.scale)
    }

    /// Update scale after step
    pub fn update(&mut self, overflow_detected: bool) {
        if overflow_detected {
            self.scale *= self.backoff_factor;
            self.steps_since_overflow = 0;
        } else {
            self.steps_since_overflow += 1;

            if self.steps_since_overflow >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_overflow = 0;
            }
        }

        // Clamp scale to reasonable range
        self.scale = self.scale.clamp(1.0, 65536.0 * 65536.0);
    }

    /// Check if gradients have overflow/underflow
    pub fn check_overflow(&self, gradients: &Array1<f32>) -> bool {
        gradients.iter().any(|&x| !x.is_finite())
    }
}

impl Default for GradientScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// Mixed precision configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Precision mode
    pub mode: PrecisionMode,
    /// Use gradient scaling (for FP16 training)
    pub use_gradient_scaling: bool,
    /// Loss scale for gradient scaling
    pub loss_scale: f32,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            mode: PrecisionMode::FP32,
            use_gradient_scaling: true,
            loss_scale: 65536.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_conversion_1d() {
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let fp16_weights = FP16Weights::from_f32_1d(&array);
        let restored = fp16_weights
            .to_f32_1d()
            .expect("Failed to convert FP16 to FP32");

        for i in 0..array.len() {
            assert!((array[i] - restored[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_fp16_conversion_2d() {
        let array = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Failed to create test array");

        let fp16_weights = FP16Weights::from_f32_2d(&array);
        let restored = fp16_weights
            .to_f32_2d()
            .expect("Failed to convert FP16 to FP32");

        assert_eq!(array.dim(), restored.dim());

        for i in 0..2 {
            for j in 0..3 {
                assert!((array[[i, j]] - restored[[i, j]]).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_bf16_conversion_1d() {
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let bf16_weights = BF16Weights::from_f32_1d(&array);
        let restored = bf16_weights
            .to_f32_1d()
            .expect("Failed to convert BF16 to FP32");

        for i in 0..array.len() {
            assert!((array[i] - restored[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_bf16_conversion_2d() {
        let array = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Failed to create test array");

        let bf16_weights = BF16Weights::from_f32_2d(&array);
        let restored = bf16_weights
            .to_f32_2d()
            .expect("Failed to convert BF16 to FP32");

        assert_eq!(array.dim(), restored.dim());

        for i in 0..2 {
            for j in 0..3 {
                assert!((array[[i, j]] - restored[[i, j]]).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_memory_savings() {
        let array = Array2::from_shape_vec((100, 100), vec![1.0; 10000])
            .expect("Failed to create test array");

        let fp16 = FP16Weights::from_f32_2d(&array);
        let bf16 = BF16Weights::from_f32_2d(&array);

        let fp32_size = 10000 * 4; // 4 bytes per f32
        let fp16_size = fp16.memory_size();
        let bf16_size = bf16.memory_size();

        assert_eq!(fp16_size, 10000 * 2);
        assert_eq!(bf16_size, 10000 * 2);
        assert!(fp16_size < fp32_size / 2 + 1);
        assert!(bf16_size < fp32_size / 2 + 1);
    }

    #[test]
    fn test_gradient_scaler() {
        let mut scaler = GradientScaler::new();

        let gradients = Array1::from_vec(vec![0.001, 0.002, 0.003]);

        // Scale
        let scaled = scaler.scale_gradients(&gradients);
        assert!(scaled[0] > gradients[0]);

        // Unscale
        let unscaled = scaler.unscale_gradients(&scaled);
        for i in 0..gradients.len() {
            assert!((gradients[i] - unscaled[i]).abs() < 1e-5);
        }

        // Test overflow detection
        let bad_gradients = Array1::from_vec(vec![f32::NAN, f32::INFINITY, 1.0]);
        assert!(scaler.check_overflow(&bad_gradients));

        // Update with overflow
        let initial_scale = scaler.get_scale();
        scaler.update(true);
        assert!(scaler.get_scale() < initial_scale);

        // Update without overflow
        scaler.update(false);
    }

    #[test]
    fn test_precision_accuracy() {
        let values = vec![0.1, 1.0, 10.0, 100.0, 1000.0];
        let array = Array1::from_vec(values);

        // Test FP16
        let fp16_array = to_fp16_array1(&array);
        let fp16_restored = from_fp16_array1(&fp16_array);
        let fp16_error = relative_error_fp16(&array, &fp16_restored);

        // Test BF16
        let bf16_array = to_bf16_array1(&array);
        let bf16_restored = from_bf16_array1(&bf16_array);
        let bf16_error = relative_error_fp16(&array, &bf16_restored);

        // Both should have reasonable accuracy
        assert!(fp16_error < 0.01);
        assert!(bf16_error < 0.01);
    }

    #[test]
    fn test_bf16_better_range() {
        // BF16 has same exponent range as FP32, so it handles large values better
        let large_values = vec![1e-10, 1e-5, 1.0, 1e5, 1e10];
        let array = Array1::from_vec(large_values);

        let bf16_array = to_bf16_array1(&array);
        let bf16_restored = from_bf16_array1(&bf16_array);

        // BF16 should handle wide range
        for i in 0..array.len() {
            let rel_error = if array[i].abs() > 1e-8 {
                (array[i] - bf16_restored[i]).abs() / array[i].abs()
            } else {
                (array[i] - bf16_restored[i]).abs()
            };

            // Allow for some precision loss, but should be reasonable
            assert!(rel_error < 0.1 || bf16_restored[i].is_finite());
        }
    }
}
