//! Common utilities and helper traits for kizzasi-tokenizer
//!
//! This module provides reusable patterns and utilities to reduce code duplication
//! across the tokenizer implementations.

use crate::error::{TokenizerError, TokenizerResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Trait for validating configuration parameters
///
/// Provides standardized validation methods that can be reused across tokenizers
/// to ensure consistent error messages and validation logic.
pub trait Validatable {
    /// Validate the configuration and return descriptive errors if invalid
    fn validate(&self) -> TokenizerResult<()>;
}

/// Helper for range validation with descriptive error messages
///
/// # Examples
/// ```
/// use kizzasi_tokenizer::utils::validate_range;
///
/// // Valid range
/// assert!(validate_range(-1.0, 1.0, "signal range").is_ok());
///
/// // Invalid range (min >= max)
/// assert!(validate_range(1.0, -1.0, "signal range").is_err());
/// ```
pub fn validate_range(min: f32, max: f32, param_name: &str) -> TokenizerResult<()> {
    if min >= max {
        return Err(TokenizerError::InvalidConfig(format!(
            "{}: min ({}) must be less than max ({})",
            param_name, min, max
        )));
    }
    if !min.is_finite() || !max.is_finite() {
        return Err(TokenizerError::InvalidConfig(format!(
            "{}: range values must be finite (got min={}, max={})",
            param_name, min, max
        )));
    }
    Ok(())
}

/// Validate that a value is within a specified range
///
/// # Arguments
/// * `value` - The value to validate
/// * `min` - Minimum allowed value (inclusive)
/// * `max` - Maximum allowed value (inclusive)
/// * `param_name` - Name of the parameter for error messages
pub fn validate_value_in_range(
    value: f32,
    min: f32,
    max: f32,
    param_name: &str,
) -> TokenizerResult<()> {
    if !(min..=max).contains(&value) {
        return Err(TokenizerError::out_of_range(value, min, max, param_name));
    }
    Ok(())
}

/// Validate that a usize value is within a specified range
///
/// # Arguments
/// * `value` - The value to validate
/// * `min` - Minimum allowed value (inclusive)
/// * `max` - Maximum allowed value (inclusive)
/// * `param_name` - Name of the parameter for error messages
pub fn validate_usize_in_range(
    value: usize,
    min: usize,
    max: usize,
    param_name: &str,
) -> TokenizerResult<()> {
    if !(min..=max).contains(&value) {
        return Err(TokenizerError::InvalidConfig(format!(
            "{}: value {} is out of range [{}..={}]",
            param_name, value, min, max
        )));
    }
    Ok(())
}

/// Validate that a dimension is positive and non-zero
pub fn validate_positive_dimension(dim: usize, param_name: &str) -> TokenizerResult<()> {
    if dim == 0 {
        return Err(TokenizerError::InvalidConfig(format!(
            "{}: dimension must be positive (got 0)",
            param_name
        )));
    }
    Ok(())
}

/// Validate that two arrays have compatible dimensions
///
/// # Arguments
/// * `arr1` - First array
/// * `arr2` - Second array
/// * `context` - Description of the operation for error messages
pub fn validate_dimensions_match(
    arr1: &Array1<f32>,
    arr2: &Array1<f32>,
    context: &str,
) -> TokenizerResult<()> {
    if arr1.len() != arr2.len() {
        return Err(TokenizerError::dim_mismatch(
            arr1.len(),
            arr2.len(),
            context,
        ));
    }
    Ok(())
}

/// Validate that a batch array has the expected shape
///
/// # Arguments
/// * `batch` - Batch array [batch_size, feature_dim]
/// * `expected_feature_dim` - Expected feature dimension
/// * `context` - Description of the operation for error messages
pub fn validate_batch_shape(
    batch: &Array2<f32>,
    expected_feature_dim: usize,
    context: &str,
) -> TokenizerResult<()> {
    if batch.ndim() != 2 {
        return Err(TokenizerError::InvalidConfig(format!(
            "{}: expected 2D batch array, got {}D",
            context,
            batch.ndim()
        )));
    }
    let actual_dim = batch.shape()[1];
    if actual_dim != expected_feature_dim {
        return Err(TokenizerError::dim_mismatch(
            expected_feature_dim,
            actual_dim,
            context,
        ));
    }
    Ok(())
}

/// Pad or truncate an array to a target length
///
/// # Arguments
/// * `array` - Input array
/// * `target_len` - Target length
/// * `pad_value` - Value to use for padding (default: 0.0)
///
/// # Returns
/// Array padded or truncated to target_len
pub fn pad_or_truncate(array: &Array1<f32>, target_len: usize, pad_value: f32) -> Array1<f32> {
    let mut result = Array1::from_elem(target_len, pad_value);
    let copy_len = array.len().min(target_len);
    result
        .slice_mut(scirs2_core::ndarray::s![..copy_len])
        .assign(&array.slice(scirs2_core::ndarray::s![..copy_len]));
    result
}

/// Clamp a value to a specified range
///
/// This is a fallible version that provides context on clamping events.
///
/// # Arguments
/// * `value` - Value to clamp
/// * `min` - Minimum value
/// * `max` - Maximum value
///
/// # Returns
/// Tuple of (clamped_value, was_clamped)
pub fn clamp_with_flag(value: f32, min: f32, max: f32) -> (f32, bool) {
    if value < min {
        (min, true)
    } else if value > max {
        (max, true)
    } else {
        (value, false)
    }
}

/// Normalize an array to [0, 1] range
///
/// # Arguments
/// * `array` - Input array
///
/// # Returns
/// Tuple of (normalized_array, min_value, max_value)
pub fn normalize_array(array: &Array1<f32>) -> TokenizerResult<(Array1<f32>, f32, f32)> {
    if array.is_empty() {
        return Err(TokenizerError::invalid_input(
            "normalize_array",
            "Cannot normalize empty array",
        ));
    }

    let min_val = array.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = array.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if !min_val.is_finite() || !max_val.is_finite() {
        return Err(TokenizerError::invalid_input(
            "normalize_array",
            "Array contains non-finite values",
        ));
    }

    let range = max_val - min_val;
    if range < 1e-8 {
        // Nearly constant array
        return Ok((Array1::zeros(array.len()), min_val, max_val));
    }

    let normalized = array.mapv(|x| (x - min_val) / range);
    Ok((normalized, min_val, max_val))
}

/// Denormalize an array from [0, 1] range back to original scale
///
/// # Arguments
/// * `array` - Normalized array
/// * `min_val` - Original minimum value
/// * `max_val` - Original maximum value
pub fn denormalize_array(array: &Array1<f32>, min_val: f32, max_val: f32) -> Array1<f32> {
    let range = max_val - min_val;
    array.mapv(|x| x * range + min_val)
}

/// Xavier/Glorot initialization for weight matrices
///
/// Initializes weights from a uniform distribution U(-limit, limit) where
/// limit = sqrt(6 / (fan_in + fan_out))
///
/// # Arguments
/// * `fan_in` - Number of input units
/// * `fan_out` - Number of output units
/// * `shape` - Shape of the weight matrix
///
/// # Returns
/// Initialized weight matrix
pub fn xavier_uniform_init(fan_in: usize, fan_out: usize, shape: (usize, usize)) -> Array2<f32> {
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let mut seed = 42u64;

    Array2::from_shape_fn(shape, |_| {
        // Simple LCG for deterministic initialization
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let val = (seed / 65536) % 32768;
        (val as f32 / 32768.0) * 2.0 * limit - limit
    })
}

/// He initialization for weight matrices (optimized for ReLU activations)
///
/// Initializes weights from a normal distribution with std = sqrt(2 / fan_in)
///
/// # Arguments
/// * `fan_in` - Number of input units
/// * `shape` - Shape of the weight matrix
///
/// # Returns
/// Initialized weight matrix
pub fn he_normal_init(fan_in: usize, shape: (usize, usize)) -> Array2<f32> {
    let std = (2.0 / fan_in as f32).sqrt();
    let mut seed = 42u64;

    Array2::from_shape_fn(shape, |_| {
        // Box-Muller transform for normal distribution
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let u1 = (seed / 65536) % 32768;
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let u2 = (seed / 65536) % 32768;

        let u1_norm = u1 as f32 / 32768.0;
        let u2_norm = u2 as f32 / 32768.0;

        let z0 = (-2.0 * u1_norm.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2_norm).cos();
        z0 * std
    })
}

/// Compute the mean of an array
pub fn array_mean(array: &Array1<f32>) -> TokenizerResult<f32> {
    if array.is_empty() {
        return Err(TokenizerError::invalid_input(
            "array_mean",
            "Cannot compute mean of empty array",
        ));
    }
    let sum: f32 = array.iter().sum();
    Ok(sum / array.len() as f32)
}

/// Compute the standard deviation of an array
pub fn array_std(array: &Array1<f32>) -> TokenizerResult<f32> {
    if array.is_empty() {
        return Err(TokenizerError::invalid_input(
            "array_std",
            "Cannot compute std of empty array",
        ));
    }
    let mean = array_mean(array)?;
    let variance = array.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / array.len() as f32;
    Ok(variance.sqrt())
}

/// Check if an array contains any non-finite values (NaN or Inf)
pub fn has_non_finite(array: &Array1<f32>) -> bool {
    array.iter().any(|&x| !x.is_finite())
}

/// Replace non-finite values with a default value
pub fn sanitize_array(array: &Array1<f32>, default_value: f32) -> Array1<f32> {
    array.mapv(|x| if x.is_finite() { x } else { default_value })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_range() {
        assert!(validate_range(-1.0, 1.0, "test").is_ok());
        assert!(validate_range(0.0, 100.0, "test").is_ok());

        // Invalid: min >= max
        assert!(validate_range(1.0, -1.0, "test").is_err());
        assert!(validate_range(5.0, 5.0, "test").is_err());

        // Invalid: non-finite
        assert!(validate_range(f32::NAN, 1.0, "test").is_err());
        assert!(validate_range(-1.0, f32::INFINITY, "test").is_err());
    }

    #[test]
    fn test_validate_value_in_range() {
        assert!(validate_value_in_range(0.5, 0.0, 1.0, "test").is_ok());
        assert!(validate_value_in_range(0.0, 0.0, 1.0, "test").is_ok());
        assert!(validate_value_in_range(1.0, 0.0, 1.0, "test").is_ok());

        assert!(validate_value_in_range(-0.1, 0.0, 1.0, "test").is_err());
        assert!(validate_value_in_range(1.1, 0.0, 1.0, "test").is_err());
    }

    #[test]
    fn test_validate_positive_dimension() {
        assert!(validate_positive_dimension(1, "test").is_ok());
        assert!(validate_positive_dimension(100, "test").is_ok());
        assert!(validate_positive_dimension(0, "test").is_err());
    }

    #[test]
    fn test_validate_dimensions_match() {
        let arr1 = Array1::zeros(10);
        let arr2 = Array1::zeros(10);
        let arr3 = Array1::zeros(20);

        assert!(validate_dimensions_match(&arr1, &arr2, "test").is_ok());
        assert!(validate_dimensions_match(&arr1, &arr3, "test").is_err());
    }

    #[test]
    fn test_pad_or_truncate() {
        let arr = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // Pad
        let padded = pad_or_truncate(&arr, 5, 0.0);
        assert_eq!(padded.len(), 5);
        assert_eq!(padded[0], 1.0);
        assert_eq!(padded[4], 0.0);

        // Truncate
        let truncated = pad_or_truncate(&arr, 2, 0.0);
        assert_eq!(truncated.len(), 2);
        assert_eq!(truncated[0], 1.0);
        assert_eq!(truncated[1], 2.0);
    }

    #[test]
    fn test_clamp_with_flag() {
        assert_eq!(clamp_with_flag(0.5, 0.0, 1.0), (0.5, false));
        assert_eq!(clamp_with_flag(-0.1, 0.0, 1.0), (0.0, true));
        assert_eq!(clamp_with_flag(1.5, 0.0, 1.0), (1.0, true));
    }

    #[test]
    fn test_normalize_array() {
        let arr = Array1::from_vec(vec![0.0, 5.0, 10.0]);
        let (normalized, min_val, max_val) = normalize_array(&arr).unwrap();

        assert_eq!(min_val, 0.0);
        assert_eq!(max_val, 10.0);
        assert!((normalized[0] - 0.0).abs() < 1e-6);
        assert!((normalized[1] - 0.5).abs() < 1e-6);
        assert!((normalized[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_denormalize_array() {
        let normalized = Array1::from_vec(vec![0.0, 0.5, 1.0]);
        let denormalized = denormalize_array(&normalized, 0.0, 10.0);

        assert!((denormalized[0] - 0.0).abs() < 1e-6);
        assert!((denormalized[1] - 5.0).abs() < 1e-6);
        assert!((denormalized[2] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_array_mean() {
        let arr = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let mean = array_mean(&arr).unwrap();
        assert!((mean - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_array_std() {
        let arr = Array1::from_vec(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let std = array_std(&arr).unwrap();
        // Expected std ≈ 2.0
        assert!((std - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_has_non_finite() {
        let arr1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(!has_non_finite(&arr1));

        let arr2 = Array1::from_vec(vec![1.0, f32::NAN, 3.0]);
        assert!(has_non_finite(&arr2));

        let arr3 = Array1::from_vec(vec![1.0, f32::INFINITY, 3.0]);
        assert!(has_non_finite(&arr3));
    }

    #[test]
    fn test_sanitize_array() {
        let arr = Array1::from_vec(vec![1.0, f32::NAN, 3.0, f32::INFINITY]);
        let sanitized = sanitize_array(&arr, 0.0);

        assert_eq!(sanitized[0], 1.0);
        assert_eq!(sanitized[1], 0.0);
        assert_eq!(sanitized[2], 3.0);
        assert_eq!(sanitized[3], 0.0);
    }

    #[test]
    fn test_xavier_uniform_init() {
        let weights = xavier_uniform_init(100, 50, (100, 50));
        assert_eq!(weights.shape(), &[100, 50]);

        // Check that values are within expected range
        let limit = (6.0 / 150.0_f32).sqrt();
        for &val in weights.iter() {
            assert!(val >= -limit && val <= limit);
        }
    }

    #[test]
    fn test_he_normal_init() {
        let weights = he_normal_init(100, (100, 50));
        assert_eq!(weights.shape(), &[100, 50]);

        // Just check that initialization doesn't panic and produces finite values
        for &val in weights.iter() {
            assert!(val.is_finite());
        }
    }
}
