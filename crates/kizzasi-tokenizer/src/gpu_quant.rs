//! GPU-accelerated quantization operations using candle
//!
//! This module provides high-performance quantization on GPU/CPU
//! using the candle tensor library for automatic device selection.
//!
//! # Features
//!
//! - Automatic GPU detection and fallback to CPU
//! - Batch quantization for maximum throughput
//! - Zero-copy tensor operations where possible
//! - Metal and CPU backends
//!
//! # Which backend actually runs
//!
//! [`auto_device`] can only return a GPU device when candle was compiled with
//! a GPU backend, which for kizzasi means this crate's off-by-default `metal`
//! feature (Apple platforms). In every other build the "GPU" quantizers here
//! run on CPU — correctly, but with no acceleration. There is deliberately no
//! CUDA path: kizzasi ships no CUDA feature at all (see `kizzasi-core`'s
//! `[features]`), so `candle_core::utils::cuda_is_available()` is a compile-
//! time `false` here and probing it would only look like a GPU option that
//! does not exist.

use crate::error::{TokenizerError, TokenizerResult};
use crate::Quantizer;
use candle_core::{Device, Tensor};

/// Select the best candle device this build can actually reach.
///
/// Returns a Metal device when the `metal` feature is enabled and a Metal GPU
/// is present, and [`Device::Cpu`] otherwise (including when Metal
/// initialisation fails, so a driver problem degrades instead of erroring).
pub fn auto_device() -> Device {
    if candle_core::utils::metal_is_available() {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    }
}

/// GPU-accelerated linear quantizer using candle
pub struct GpuLinearQuantizer {
    /// Minimum value of range
    min: f32,
    /// Maximum value of range
    max: f32,
    /// Number of quantization levels
    levels: usize,
    /// Candle device (GPU or CPU)
    device: Device,
}

impl GpuLinearQuantizer {
    /// Create a new GPU-accelerated quantizer
    ///
    /// Automatically detects and uses GPU if available, falls back to CPU.
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum value of quantization range
    /// * `max` - Maximum value of quantization range
    /// * `bits` - Number of bits for quantization
    pub fn new(min: f32, max: f32, bits: u8) -> TokenizerResult<Self> {
        if min >= max {
            return Err(TokenizerError::InvalidConfig(
                "min must be less than max".into(),
            ));
        }
        if bits == 0 || bits > 16 {
            return Err(TokenizerError::InvalidConfig("bits must be 1-16".into()));
        }

        let levels = 1usize << bits;

        // Metal when this build can reach it, CPU otherwise.
        let device = auto_device();

        Ok(Self {
            min,
            max,
            levels,
            device,
        })
    }

    /// Create quantizer with explicit device selection
    pub fn with_device(min: f32, max: f32, bits: u8, device: Device) -> TokenizerResult<Self> {
        if min >= max {
            return Err(TokenizerError::InvalidConfig(
                "min must be less than max".into(),
            ));
        }
        if bits == 0 || bits > 16 {
            return Err(TokenizerError::InvalidConfig("bits must be 1-16".into()));
        }

        let levels = 1usize << bits;

        Ok(Self {
            min,
            max,
            levels,
            device,
        })
    }

    /// Quantize a batch of signals on GPU
    ///
    /// # Arguments
    ///
    /// * `signals` - Slice of signals to quantize
    ///
    /// # Returns
    ///
    /// Quantized levels as `Vec<i32>`
    pub fn quantize_batch(&self, signals: &[f32]) -> TokenizerResult<Vec<i32>> {
        // Convert to tensor on device
        let tensor = Tensor::from_slice(signals, signals.len(), &self.device).map_err(|e| {
            TokenizerError::encoding("serialization", format!("Tensor creation failed: {}", e))
        })?;

        // Clamp to range
        let clamped = tensor
            .clamp(self.min as f64, self.max as f64)
            .map_err(|e| {
                TokenizerError::encoding("serialization", format!("Clamp failed: {}", e))
            })?;

        // Normalize to [0, 1]
        let range = self.max - self.min;
        let normalized = ((clamped - self.min as f64).map_err(|e| {
            TokenizerError::encoding("serialization", format!("Subtraction failed: {}", e))
        })? / range as f64)
            .map_err(|e| {
                TokenizerError::encoding("serialization", format!("Division failed: {}", e))
            })?;

        // Scale to levels and round
        let scaled = (normalized * (self.levels - 1) as f64).map_err(|e| {
            TokenizerError::encoding("serialization", format!("Scaling failed: {}", e))
        })?;

        let rounded = scaled.round().map_err(|e| {
            TokenizerError::encoding("serialization", format!("Rounding failed: {}", e))
        })?;

        // Convert to CPU and extract
        let cpu_tensor = rounded.to_device(&Device::Cpu).map_err(|e| {
            TokenizerError::encoding("serialization", format!("Transfer to CPU failed: {}", e))
        })?;

        let result_f32: Vec<f32> = cpu_tensor.to_vec1().map_err(|e| {
            TokenizerError::encoding("serialization", format!("Tensor to vec failed: {}", e))
        })?;

        Ok(result_f32.into_iter().map(|v| v as i32).collect())
    }

    /// Dequantize a batch of quantized levels on GPU
    pub fn dequantize_batch(&self, levels: &[i32]) -> TokenizerResult<Vec<f32>> {
        // Convert i32 to f32 for tensor
        let levels_f32: Vec<f32> = levels.iter().map(|&l| l as f32).collect();

        // Create tensor on device
        let tensor =
            Tensor::from_slice(&levels_f32, levels_f32.len(), &self.device).map_err(|e| {
                TokenizerError::decoding(
                    "deserialization",
                    format!("Tensor creation failed: {}", e),
                )
            })?;

        // Clamp levels
        let max_level = (self.levels - 1) as f32;
        let clamped = tensor.clamp(0.0, max_level as f64).map_err(|e| {
            TokenizerError::decoding("deserialization", format!("Clamp failed: {}", e))
        })?;

        // Normalize to [0, 1]
        let normalized = (clamped / max_level as f64).map_err(|e| {
            TokenizerError::decoding("deserialization", format!("Division failed: {}", e))
        })?;

        // Scale to range
        let range = self.max - self.min;
        let result = ((normalized * range as f64).map_err(|e| {
            TokenizerError::decoding("deserialization", format!("Scaling failed: {}", e))
        })? + self.min as f64)
            .map_err(|e| {
                TokenizerError::decoding("deserialization", format!("Addition failed: {}", e))
            })?;

        // Transfer to CPU and extract
        let cpu_tensor = result.to_device(&Device::Cpu).map_err(|e| {
            TokenizerError::decoding("deserialization", format!("Transfer to CPU failed: {}", e))
        })?;

        cpu_tensor.to_vec1().map_err(|e| {
            TokenizerError::decoding("deserialization", format!("Tensor to vec failed: {}", e))
        })
    }

    /// Get the device being used
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if using GPU
    ///
    /// `Device::Cuda` is matched for completeness -- `candle_core::Device`
    /// carries the variant on every build -- but kizzasi never constructs one:
    /// only the `metal` feature can produce a GPU device here.
    pub fn is_gpu(&self) -> bool {
        matches!(self.device, Device::Cuda(_) | Device::Metal(_))
    }
}

impl Quantizer for GpuLinearQuantizer {
    fn quantize(&self, value: f32) -> i32 {
        // Single value quantization (less efficient, prefer batch)
        let clamped = value.clamp(self.min, self.max);
        let normalized = (clamped - self.min) / (self.max - self.min);
        (normalized * (self.levels - 1) as f32).round() as i32
    }

    fn dequantize(&self, level: i32) -> f32 {
        let clamped_level = level.clamp(0, (self.levels - 1) as i32);
        let normalized = clamped_level as f32 / (self.levels - 1) as f32;
        self.min + normalized * (self.max - self.min)
    }

    fn num_levels(&self) -> usize {
        self.levels
    }
}

/// GPU-accelerated Vector Quantizer for batched operations
pub struct GpuVectorQuantizer {
    /// Codebook of shape (codebook_size, vector_dim)
    codebook: Tensor,
    /// Vector dimension
    vector_dim: usize,
    /// Codebook size
    codebook_size: usize,
    /// Device
    device: Device,
}

impl GpuVectorQuantizer {
    /// Create a new GPU vector quantizer
    ///
    /// # Arguments
    ///
    /// * `codebook_size` - Number of codebook vectors
    /// * `vector_dim` - Dimension of each vector
    pub fn new(codebook_size: usize, vector_dim: usize) -> TokenizerResult<Self> {
        // Auto-select device (Metal when reachable, CPU otherwise).
        let device = auto_device();

        // Random initialization
        let codebook = Tensor::randn(0.0f32, 1.0f32, (codebook_size, vector_dim), &device)
            .map_err(|e| {
                TokenizerError::InvalidConfig(format!("Codebook initialization failed: {}", e))
            })?;

        Ok(Self {
            codebook,
            vector_dim,
            codebook_size,
            device,
        })
    }

    /// Quantize vectors to nearest codebook entries
    ///
    /// # Arguments
    ///
    /// * `vectors` - Input vectors of shape (batch_size, vector_dim)
    ///
    /// # Returns
    ///
    /// Codebook indices for each vector
    pub fn quantize_vectors(
        &self,
        vectors: &[f32],
        batch_size: usize,
    ) -> TokenizerResult<Vec<usize>> {
        if vectors.len() != batch_size * self.vector_dim {
            return Err(TokenizerError::dim_mismatch(
                batch_size * self.vector_dim,
                vectors.len(),
                "dimension validation",
            ));
        }

        // Create input tensor [batch_size, vector_dim]
        let input = Tensor::from_slice(vectors, (batch_size, self.vector_dim), &self.device)
            .map_err(|e| {
                TokenizerError::encoding("serialization", format!("Input tensor creation: {}", e))
            })?;

        // Compute distances: ||x - c_i||^2 = ||x||^2 + ||c_i||^2 - 2 * x · c_i
        // input: [batch, dim]
        // codebook: [codebook_size, dim]

        // ||x||^2 for each input vector [batch]
        let input_norm = input
            .sqr()
            .map_err(|e| TokenizerError::encoding("serialization", e.to_string()))?
            .sum_keepdim(1)
            .map_err(|e| TokenizerError::encoding("serialization", e.to_string()))?;

        // ||c_i||^2 for each codebook vector [codebook_size]
        let codebook_norm = self
            .codebook
            .sqr()
            .map_err(|e| TokenizerError::encoding("serialization", e.to_string()))?
            .sum_keepdim(1)
            .map_err(|e| TokenizerError::encoding("serialization", e.to_string()))?
            .t()
            .map_err(|e| TokenizerError::encoding("serialization", e.to_string()))?;

        // x · c_i^T: [batch, codebook_size]
        let dot_product = input
            .matmul(
                &self
                    .codebook
                    .t()
                    .map_err(|e| TokenizerError::encoding("serialization", e.to_string()))?,
            )
            .map_err(|e| TokenizerError::encoding("serialization", e.to_string()))?;

        // distances = ||x||^2 + ||c||^2 - 2 * x·c
        let distances = (input_norm
            .broadcast_add(&codebook_norm)
            .map_err(|e| TokenizerError::encoding("serialization", e.to_string()))?
            - (dot_product * 2.0)
                .map_err(|e| TokenizerError::encoding("serialization", e.to_string()))?)
        .map_err(|e| TokenizerError::encoding("serialization", e.to_string()))?;

        // Find argmin for each row
        let indices = distances
            .argmin_keepdim(1)
            .map_err(|e| TokenizerError::encoding("serialization", e.to_string()))?
            .squeeze(1)
            .map_err(|e| TokenizerError::encoding("serialization", e.to_string()))?;

        // Convert to CPU and extract
        let cpu_indices = indices.to_device(&Device::Cpu).map_err(|e| {
            TokenizerError::encoding("serialization", format!("CPU transfer: {}", e))
        })?;

        let indices_u32: Vec<u32> = cpu_indices.to_vec1().map_err(|e| {
            TokenizerError::encoding("serialization", format!("Vec conversion: {}", e))
        })?;

        Ok(indices_u32.into_iter().map(|i| i as usize).collect())
    }

    /// Dequantize indices back to vectors
    pub fn dequantize_vectors(&self, indices: &[usize]) -> TokenizerResult<Vec<f32>> {
        // Gather codebook vectors
        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let indices_tensor = Tensor::from_slice(&indices_u32, indices_u32.len(), &self.device)
            .map_err(|e| {
                TokenizerError::decoding("deserialization", format!("Index tensor creation: {}", e))
            })?;

        // Index select from codebook
        let result = self
            .codebook
            .index_select(&indices_tensor, 0)
            .map_err(|e| {
                TokenizerError::decoding("deserialization", format!("Index select: {}", e))
            })?;

        // Transfer to CPU
        let cpu_result = result.to_device(&Device::Cpu).map_err(|e| {
            TokenizerError::decoding("deserialization", format!("CPU transfer: {}", e))
        })?;

        // Flatten to vec
        let result_vec: Vec<f32> = cpu_result
            .to_vec1()
            .or_else(|_| {
                // Try flatten if multi-dimensional
                cpu_result.flatten_all().and_then(|t| t.to_vec1())
            })
            .map_err(|e| {
                TokenizerError::decoding("deserialization", format!("Vec conversion: {}", e))
            })?;

        Ok(result_vec)
    }

    /// Update codebook (for training)
    pub fn update_codebook(&mut self, new_codebook: Tensor) -> TokenizerResult<()> {
        if new_codebook.shape().dims() != [self.codebook_size, self.vector_dim] {
            return Err(TokenizerError::InvalidConfig(
                "Codebook shape mismatch".into(),
            ));
        }

        self.codebook = new_codebook;
        Ok(())
    }

    /// Get reference to codebook tensor
    pub fn codebook(&self) -> &Tensor {
        &self.codebook
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_quantizer_creation() {
        let quantizer = GpuLinearQuantizer::new(-1.0, 1.0, 8).unwrap();
        assert_eq!(quantizer.num_levels(), 256);
    }

    #[test]
    fn test_gpu_quantizer_single_value() {
        let quantizer = GpuLinearQuantizer::new(-1.0, 1.0, 8).unwrap();

        let level = quantizer.quantize(0.0);
        assert!((level - 127).abs() <= 1);

        let value = quantizer.dequantize(127);
        assert!(value.abs() < 0.01);
    }

    #[test]
    fn test_gpu_quantizer_batch() {
        let quantizer = GpuLinearQuantizer::new(-1.0, 1.0, 8).unwrap();

        let signals = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let quantized = quantizer.quantize_batch(&signals).unwrap();

        assert_eq!(quantized.len(), signals.len());

        // Middle value should be around 127
        assert!((quantized[0] - 127).abs() <= 2);

        // Extremes
        assert!(quantized[2] >= 250); // 1.0
        assert!(quantized[4] <= 5); // -1.0
    }

    #[test]
    fn test_gpu_quantizer_roundtrip() {
        let quantizer = GpuLinearQuantizer::new(-10.0, 10.0, 10).unwrap();

        let signals: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 5.0).collect();

        let quantized = quantizer.quantize_batch(&signals).unwrap();
        let dequantized = quantizer.dequantize_batch(&quantized).unwrap();

        for i in 0..signals.len() {
            assert!(
                (signals[i] - dequantized[i]).abs() < 0.1,
                "Mismatch at {}: {} vs {}",
                i,
                signals[i],
                dequantized[i]
            );
        }
    }

    #[test]
    fn test_gpu_vector_quantizer() {
        let quantizer = GpuVectorQuantizer::new(16, 4).unwrap();

        // Create some test vectors [batch=3, dim=4]
        let vectors = vec![
            1.0, 2.0, 3.0, 4.0, // vector 1
            5.0, 6.0, 7.0, 8.0, // vector 2
            1.1, 2.1, 3.1, 4.1, // vector 3 (similar to 1)
        ];

        let indices = quantizer.quantize_vectors(&vectors, 3).unwrap();
        assert_eq!(indices.len(), 3);

        // Similar vectors should have similar or same indices
        let dequantized = quantizer.dequantize_vectors(&indices).unwrap();
        assert_eq!(dequantized.len(), 12); // 3 vectors * 4 dims
    }

    #[test]
    fn test_gpu_vector_quantizer_roundtrip() {
        let quantizer = GpuVectorQuantizer::new(32, 8).unwrap();

        let batch_size = 10;
        let vectors: Vec<f32> = (0..batch_size * 8).map(|i| i as f32 * 0.1).collect();

        let indices = quantizer.quantize_vectors(&vectors, batch_size).unwrap();
        let reconstructed = quantizer.dequantize_vectors(&indices).unwrap();

        assert_eq!(reconstructed.len(), vectors.len());

        // Reconstruction should be reasonably close
        let mse: f32 = vectors
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / vectors.len() as f32;

        // MSE should be finite
        assert!(mse.is_finite());
    }

    #[test]
    fn test_device_selection() {
        let quantizer = GpuLinearQuantizer::new(-1.0, 1.0, 8).unwrap();

        // Device should be valid (test passes if device() returns successfully)
        let _device = quantizer.device();
        // No assertion needed - if device() returns, it's one of the valid variants
    }
}
