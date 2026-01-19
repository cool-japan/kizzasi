//! GPU memory management and tensor transfer utilities
//!
//! Provides efficient utilities for:
//! - Moving tensors between CPU and GPU
//! - Memory pooling for GPU tensors
//! - Batch transfer operations
//! - Memory usage tracking
//!
//! # Examples
//!
//! ```rust
//! use kizzasi_core::gpu_utils::{TensorTransfer, TransferBatch};
//! use kizzasi_core::device::DeviceConfig;
//! use candle_core::{Tensor, Device};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Transfer single tensor to GPU
//! let cpu_tensor = Tensor::zeros((100, 100), candle_core::DType::F32, &Device::Cpu)?;
//! let gpu_device = DeviceConfig::default().create_device()?;
//!
//! let gpu_tensor = TensorTransfer::to_device(&cpu_tensor, &gpu_device)?;
//!
//! // Batch transfer multiple tensors
//! let tensors = vec![cpu_tensor.clone(), cpu_tensor.clone()];
//! let gpu_tensors = TransferBatch::transfer_all(&tensors, &gpu_device)?;
//! # Ok(())
//! # }
//! ```

use crate::error::{CoreError, CoreResult};
use candle_core::{Device, Tensor};
use std::collections::HashMap;

/// Tensor transfer utilities for CPU/GPU operations
pub struct TensorTransfer;

impl TensorTransfer {
    /// Transfer a tensor to a specific device
    ///
    /// # Arguments
    /// * `tensor` - The tensor to transfer
    /// * `device` - Target device
    ///
    /// # Returns
    /// Tensor on the target device
    pub fn to_device(tensor: &Tensor, device: &Device) -> CoreResult<Tensor> {
        tensor
            .to_device(device)
            .map_err(|e| CoreError::DeviceError(format!("Failed to transfer tensor: {}", e)))
    }

    /// Transfer a tensor to CPU
    pub fn to_cpu(tensor: &Tensor) -> CoreResult<Tensor> {
        Self::to_device(tensor, &Device::Cpu)
    }

    /// Transfer a tensor to GPU (auto-detect best GPU)
    pub fn to_gpu(tensor: &Tensor) -> CoreResult<Tensor> {
        let device = crate::device::get_best_device();
        if matches!(device, Device::Cpu) {
            return Err(CoreError::DeviceError(
                "No GPU device available".to_string(),
            ));
        }
        Self::to_device(tensor, &device)
    }

    /// Check if a tensor is on GPU
    pub fn is_on_gpu(tensor: &Tensor) -> bool {
        !matches!(tensor.device(), Device::Cpu)
    }

    /// Check if a tensor is on CPU
    pub fn is_on_cpu(tensor: &Tensor) -> bool {
        matches!(tensor.device(), Device::Cpu)
    }

    /// Get device of a tensor
    pub fn get_device(tensor: &Tensor) -> Device {
        tensor.device().clone()
    }
}

/// Batch tensor transfer operations
pub struct TransferBatch;

impl TransferBatch {
    /// Transfer multiple tensors to a device in batch
    ///
    /// This is more efficient than transferring tensors one by one
    /// as it can leverage async transfers on some backends.
    pub fn transfer_all(tensors: &[Tensor], device: &Device) -> CoreResult<Vec<Tensor>> {
        tensors
            .iter()
            .map(|t| TensorTransfer::to_device(t, device))
            .collect()
    }

    /// Transfer all tensors to CPU
    pub fn to_cpu_all(tensors: &[Tensor]) -> CoreResult<Vec<Tensor>> {
        Self::transfer_all(tensors, &Device::Cpu)
    }

    /// Transfer all tensors to GPU
    pub fn to_gpu_all(tensors: &[Tensor]) -> CoreResult<Vec<Tensor>> {
        let device = crate::device::get_best_device();
        if matches!(device, Device::Cpu) {
            return Err(CoreError::DeviceError(
                "No GPU device available".to_string(),
            ));
        }
        Self::transfer_all(tensors, &device)
    }
}

/// Memory usage tracking for GPU tensors
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total memory allocated (bytes)
    pub total_allocated: usize,
    /// Number of tensors tracked
    pub tensor_count: usize,
    /// Memory by tensor name
    pub memory_by_name: HashMap<String, usize>,
}

impl MemoryStats {
    /// Create a new memory stats tracker
    pub fn new() -> Self {
        Self {
            total_allocated: 0,
            tensor_count: 0,
            memory_by_name: HashMap::new(),
        }
    }

    /// Track a tensor's memory usage
    pub fn track_tensor(&mut self, name: String, tensor: &Tensor) {
        let size = Self::tensor_size(tensor);
        self.total_allocated += size;
        self.tensor_count += 1;
        self.memory_by_name.insert(name, size);
    }

    /// Untrack a tensor
    pub fn untrack_tensor(&mut self, name: &str) {
        if let Some(size) = self.memory_by_name.remove(name) {
            self.total_allocated = self.total_allocated.saturating_sub(size);
            self.tensor_count = self.tensor_count.saturating_sub(1);
        }
    }

    /// Get total allocated memory in bytes
    pub fn total_bytes(&self) -> usize {
        self.total_allocated
    }

    /// Get total allocated memory in MB
    pub fn total_mb(&self) -> f64 {
        self.total_allocated as f64 / (1024.0 * 1024.0)
    }

    /// Get total allocated memory in GB
    pub fn total_gb(&self) -> f64 {
        self.total_allocated as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Calculate tensor size in bytes
    fn tensor_size(tensor: &Tensor) -> usize {
        let elem_count: usize = tensor.dims().iter().product();
        let dtype_size = match tensor.dtype() {
            candle_core::DType::U8 => 1,
            candle_core::DType::U32 => 4,
            candle_core::DType::I64 => 8,
            candle_core::DType::F16 => 2,
            candle_core::DType::BF16 => 2,
            candle_core::DType::F32 => 4,
            candle_core::DType::F64 => 8,
        };
        elem_count * dtype_size
    }

    /// Clear all tracked tensors
    pub fn clear(&mut self) {
        self.total_allocated = 0;
        self.tensor_count = 0;
        self.memory_by_name.clear();
    }
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU memory pool for efficient tensor allocation
pub struct GPUMemoryPool {
    device: Device,
    stats: MemoryStats,
}

impl GPUMemoryPool {
    /// Create a new GPU memory pool
    pub fn new(device: Device) -> Self {
        Self {
            device,
            stats: MemoryStats::new(),
        }
    }

    /// Allocate a tensor on GPU
    pub fn allocate(
        &mut self,
        name: String,
        shape: &[usize],
        dtype: candle_core::DType,
    ) -> CoreResult<Tensor> {
        let tensor = Tensor::zeros(shape, dtype, &self.device)
            .map_err(|e| CoreError::DeviceError(format!("Failed to allocate tensor: {}", e)))?;

        self.stats.track_tensor(name, &tensor);
        Ok(tensor)
    }

    /// Release a tensor from the pool
    pub fn release(&mut self, name: &str) {
        self.stats.untrack_tensor(name);
    }

    /// Get memory statistics
    pub fn stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Prefetching utilities for optimizing data transfer
pub struct TensorPrefetch;

impl TensorPrefetch {
    /// Prefetch tensor to device (async hint for backends that support it)
    ///
    /// Note: This is a hint to the backend. Actual async behavior depends on
    /// the device backend (CUDA/Metal).
    pub fn prefetch(tensor: &Tensor, device: &Device) -> CoreResult<Tensor> {
        // For now, this is synchronous. Future implementations could leverage
        // async streams on CUDA or Metal command buffers
        TensorTransfer::to_device(tensor, device)
    }

    /// Prefetch multiple tensors
    pub fn prefetch_batch(tensors: &[Tensor], device: &Device) -> CoreResult<Vec<Tensor>> {
        TransferBatch::transfer_all(tensors, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_tensor_transfer_to_cpu() {
        let tensor = Tensor::zeros((10, 10), DType::F32, &Device::Cpu).unwrap();
        let cpu_tensor = TensorTransfer::to_cpu(&tensor).unwrap();

        assert!(TensorTransfer::is_on_cpu(&cpu_tensor));
        assert!(!TensorTransfer::is_on_gpu(&cpu_tensor));
    }

    #[test]
    fn test_batch_transfer() {
        let tensors = vec![
            Tensor::zeros((5, 5), DType::F32, &Device::Cpu).unwrap(),
            Tensor::zeros((10, 10), DType::F32, &Device::Cpu).unwrap(),
        ];

        let cpu_tensors = TransferBatch::to_cpu_all(&tensors).unwrap();
        assert_eq!(cpu_tensors.len(), 2);

        for tensor in &cpu_tensors {
            assert!(TensorTransfer::is_on_cpu(tensor));
        }
    }

    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::new();

        let tensor1 = Tensor::zeros((10, 10), DType::F32, &Device::Cpu).unwrap();
        let tensor2 = Tensor::zeros((20, 20), DType::F32, &Device::Cpu).unwrap();

        stats.track_tensor("tensor1".to_string(), &tensor1);
        stats.track_tensor("tensor2".to_string(), &tensor2);

        assert_eq!(stats.tensor_count, 2);
        // 10*10*4 + 20*20*4 = 400 + 1600 = 2000 bytes
        assert_eq!(stats.total_bytes(), 2000);

        stats.untrack_tensor("tensor1");
        assert_eq!(stats.tensor_count, 1);
        assert_eq!(stats.total_bytes(), 1600);

        stats.clear();
        assert_eq!(stats.tensor_count, 0);
        assert_eq!(stats.total_bytes(), 0);
    }

    #[test]
    fn test_memory_stats_mb_gb() {
        let mut stats = MemoryStats::new();

        // Create a large tensor: 1000 * 1000 * f32 (4 bytes) = 4,000,000 bytes
        let tensor = Tensor::zeros((1000, 1000), DType::F32, &Device::Cpu).unwrap();
        stats.track_tensor("large_tensor".to_string(), &tensor);

        // 4,000,000 bytes = 3.814 MB (1024^2) or 4.0 MB (1000^2)
        // Using 1024-based calculation: 4,000,000 / (1024 * 1024) ≈ 3.814 MB
        let expected_mb = 4_000_000.0 / (1024.0 * 1024.0);
        assert!((stats.total_mb() - expected_mb).abs() < 0.01);

        let expected_gb = 4_000_000.0 / (1024.0 * 1024.0 * 1024.0);
        assert!((stats.total_gb() - expected_gb).abs() < 0.0001);
    }

    #[test]
    fn test_gpu_memory_pool() {
        let mut pool = GPUMemoryPool::new(Device::Cpu);

        let tensor = pool
            .allocate("test_tensor".to_string(), &[100, 100], DType::F32)
            .unwrap();

        assert_eq!(tensor.dims(), &[100, 100]);
        assert_eq!(pool.stats().tensor_count, 1);
        assert_eq!(pool.stats().total_bytes(), 100 * 100 * 4);

        pool.release("test_tensor");
        assert_eq!(pool.stats().tensor_count, 0);
    }

    #[test]
    fn test_get_device() {
        let tensor = Tensor::zeros((10, 10), DType::F32, &Device::Cpu).unwrap();
        let device = TensorTransfer::get_device(&tensor);
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_tensor_size_calculation() {
        let tensor_f32 = Tensor::zeros((10, 20), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(MemoryStats::tensor_size(&tensor_f32), 10 * 20 * 4);

        let tensor_f16 = Tensor::zeros((10, 20), DType::F16, &Device::Cpu).unwrap();
        assert_eq!(MemoryStats::tensor_size(&tensor_f16), 10 * 20 * 2);

        let tensor_i64 = Tensor::zeros((5, 5), DType::I64, &Device::Cpu).unwrap();
        assert_eq!(MemoryStats::tensor_size(&tensor_i64), 5 * 5 * 8);
    }
}
