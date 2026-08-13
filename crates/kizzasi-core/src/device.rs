//! Device selection and GPU acceleration utilities
//!
//! Provides a device-selection API (Metal/CPU) and device management for
//! efficient model training and inference.
//!
//! # GPU status
//!
//! - **CPU** — always available.
//! - **Metal** — real, behind the off-by-default `metal` Cargo feature, which
//!   forwards to candle's own Metal backend through the `kizzasi-metal` shim.
//!   With `metal` enabled on an Apple machine, [`DeviceType::Metal`] creates a
//!   live GPU device, [`is_metal_available`] reports `true` and
//!   [`get_best_device`] returns it. Without the feature the
//!   [`DeviceType::Metal`] variant does not exist, so there is no code path
//!   that can silently pretend a GPU is present.
//!
//!   Enabling `metal` on a *non*-Apple target (which is what `--all-features`
//!   does on Linux and Windows) is buildable but inert, because Cargo features
//!   are not target-aware and candle's Metal backend does not compile off
//!   Apple — see the `kizzasi-metal` crate docs. Inert means loud, not silent:
//!   the [`DeviceType::Metal`] variant exists, but [`is_metal_available`]
//!   returns `false`, [`get_best_device`] stays on CPU, and
//!   [`DeviceConfig::create_device`] returns a [`CoreError::DeviceError`]
//!   naming the target instead of handing back a CPU device in disguise.
//! - **CUDA** — not offered. There is no `cuda` Cargo feature and no
//!   `DeviceType::Cuda` variant: candle's CUDA backend requires an installed
//!   NVIDIA toolkit at *build* time (its build scripts abort without one), and
//!   Cargo cannot make a feature conditional on the host toolchain, so such a
//!   flag would break `--all-features` builds everywhere else. See
//!   `Cargo.toml` for the full rationale, including the target-scoped
//!   dependency-alias workaround that Cargo's own `cargo metadata` rejects.
//!   For portable GPU acceleration use the `kizzasi-webgpu` backend (the
//!   `webgpu` feature of the `kizzasi` facade crate).
//!
//! # Features
//!
//! - **Auto-detection**: Tries Metal (when compiled in), then falls back to CPU
//! - **Fallback**: Gracefully falls back to CPU if GPU is unavailable
//! - **Memory Management**: Utilities for efficient GPU memory usage
//! - **Multi-GPU**: Support for selecting specific GPU devices
//!
//! # Examples
//!
//! ```rust
//! use kizzasi_core::device::{DeviceConfig, DeviceType, get_best_device};
//!
//! // Auto-select best available device
//! let device = get_best_device();
//!
//! // Or configure manually
//! let config = DeviceConfig::default()
//!     .with_device_type(DeviceType::Cpu)
//!     .with_device_id(0);
//! let device = config.create_device()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#[cfg(feature = "metal")]
use crate::error::CoreError;
use crate::error::CoreResult;
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Device type for model execution
///
/// `Metal` only exists when the `metal` feature is enabled, so a build that
/// cannot reach a GPU cannot even name one. There is no `Cuda` variant — see
/// the module documentation for why CUDA is not offered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// CPU execution (always available)
    Cpu,
    /// Apple Metal GPU (requires the `metal` feature, Apple platforms only)
    #[cfg(feature = "metal")]
    Metal,
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "CPU"),
            #[cfg(feature = "metal")]
            DeviceType::Metal => write!(f, "Metal"),
        }
    }
}

/// Device configuration for GPU acceleration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Device type to use
    pub device_type: DeviceType,
    /// Device ID (for multi-GPU systems)
    pub device_id: usize,
    /// Enable mixed precision (FP16)
    pub use_fp16: bool,
    /// Requested TF32 matmul. Recorded only -- see [`DeviceConfig::with_tf32`];
    /// TF32 is a CUDA tensor-core setting and kizzasi ships no CUDA backend.
    pub use_tf32: bool,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Cpu,
            device_id: 0,
            use_fp16: false,
            use_tf32: false,
        }
    }
}

impl DeviceConfig {
    /// Create a new device configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set device type
    pub fn with_device_type(mut self, device_type: DeviceType) -> Self {
        self.device_type = device_type;
        self
    }

    /// Set device ID
    pub fn with_device_id(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }

    /// Enable FP16 precision
    pub fn with_fp16(mut self, enabled: bool) -> Self {
        self.use_fp16 = enabled;
        self
    }

    /// Enable TF32 precision
    ///
    /// Recorded for forward compatibility only: TF32 is a CUDA-tensor-core
    /// setting and kizzasi ships no CUDA backend, so no backend reads this
    /// flag today.
    pub fn with_tf32(mut self, enabled: bool) -> Self {
        self.use_tf32 = enabled;
        self
    }

    /// Create a candle Device from this configuration
    pub fn create_device(&self) -> CoreResult<Device> {
        match self.device_type {
            DeviceType::Cpu => Ok(Device::Cpu),

            #[cfg(feature = "metal")]
            DeviceType::Metal => kizzasi_metal::new_device(self.device_id).map_err(|e| {
                CoreError::DeviceError(format!(
                    "Failed to create Metal device {}: {}",
                    self.device_id, e
                ))
            }),
        }
    }
}

/// Check if Metal is available
///
/// `false` whenever no Metal device can be opened — including a build that
/// enabled the `metal` feature on a non-Apple target, where the backend was
/// never compiled in.
pub fn is_metal_available() -> bool {
    #[cfg(feature = "metal")]
    {
        kizzasi_metal::is_available()
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

/// Get the best available device (Metal > CPU)
pub fn get_best_device() -> Device {
    #[cfg(feature = "metal")]
    {
        match kizzasi_metal::new_device(0) {
            Ok(device) => {
                tracing::info!("Using Metal device 0");
                return device;
            }
            Err(e) => tracing::debug!("Metal device 0 unavailable, falling back to CPU: {e}"),
        }
    }

    tracing::info!("Using CPU device");
    Device::Cpu
}

/// Get available Metal devices
///
/// Only device 0 is probed: candle's Metal backend indexes an internal `Vec`
/// without bounds-checking on some paths, so walking ordinals upward can panic
/// rather than return an error on multi-GPU Macs.
#[cfg(feature = "metal")]
pub fn get_metal_devices() -> Vec<usize> {
    kizzasi_metal::device_ordinals()
}

/// Device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device type
    pub device_type: DeviceType,
    /// Device ID
    pub device_id: usize,
    /// Device name (if available)
    pub name: Option<String>,
    /// Total memory (bytes, if available)
    pub total_memory: Option<u64>,
    /// Available memory (bytes, if available)
    pub available_memory: Option<u64>,
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} Device {}", self.device_type, self.device_id)?;
        if let Some(name) = &self.name {
            write!(f, " ({})", name)?;
        }
        if let Some(total) = self.total_memory {
            write!(f, " - Total Memory: {} GB", total / (1024 * 1024 * 1024))?;
        }
        if let Some(available) = self.available_memory {
            write!(f, " - Available: {} GB", available / (1024 * 1024 * 1024))?;
        }
        Ok(())
    }
}

/// Get information about a device
pub fn get_device_info(device: &Device) -> DeviceInfo {
    match device {
        Device::Cpu => DeviceInfo {
            device_type: DeviceType::Cpu,
            device_id: 0,
            name: Some("CPU".to_string()),
            total_memory: None,
            available_memory: None,
        },

        #[cfg(feature = "metal")]
        Device::Metal(_metal_device) => {
            DeviceInfo {
                device_type: DeviceType::Metal,
                device_id: 0,           // Metal devices are numbered sequentially
                name: None,             // Could query via Metal API
                total_memory: None,     // Could query via Metal API
                available_memory: None, // Could query via Metal API
            }
        }

        // Catch-all for candle device variants this build cannot describe:
        // `candle_core::Device` always carries every variant regardless of the
        // backend features compiled in, so a `Device::Metal` handed to a build
        // without the `metal` feature lands here rather than being reported as
        // a device type that does not exist in this build's `DeviceType`.
        #[allow(unreachable_patterns)]
        _ => DeviceInfo {
            device_type: DeviceType::Cpu,
            device_id: 0,
            name: Some("Unknown".to_string()),
            total_memory: None,
            available_memory: None,
        },
    }
}

/// List all available devices
pub fn list_devices() -> Vec<DeviceInfo> {
    #[allow(unused_mut)]
    let mut result = vec![DeviceInfo {
        device_type: DeviceType::Cpu,
        device_id: 0,
        name: Some("CPU".to_string()),
        total_memory: None,
        available_memory: None,
    }];

    #[cfg(feature = "metal")]
    {
        for id in get_metal_devices() {
            if let Ok(device) = kizzasi_metal::new_device(id) {
                result.push(get_device_info(&device));
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_config_default() {
        let config = DeviceConfig::default();
        assert_eq!(config.device_type, DeviceType::Cpu);
        assert_eq!(config.device_id, 0);
        assert!(!config.use_fp16);
        assert!(!config.use_tf32);
    }

    #[test]
    fn test_device_config_builder() {
        let config = DeviceConfig::new()
            .with_device_id(1)
            .with_fp16(true)
            .with_tf32(true);

        assert_eq!(config.device_id, 1);
        assert!(config.use_fp16);
        assert!(config.use_tf32);
    }

    #[test]
    fn test_cpu_device_creation() {
        let config = DeviceConfig::new();
        let device = config.create_device().unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_get_best_device() {
        let device = get_best_device();
        // Should always succeed - just check that we got a valid device
        // (CPU, or Metal when the `metal` feature is on and a GPU is present)
        let _ = device; // Valid device was created
    }

    #[test]
    fn test_list_devices() {
        let devices = list_devices();
        // Should always have at least CPU
        assert!(!devices.is_empty());
        assert_eq!(devices[0].device_type, DeviceType::Cpu);
    }

    #[test]
    fn test_device_info_display() {
        let info = DeviceInfo {
            device_type: DeviceType::Cpu,
            device_id: 0,
            name: Some("Test CPU".to_string()),
            total_memory: Some(16 * 1024 * 1024 * 1024), // 16 GB
            available_memory: Some(8 * 1024 * 1024 * 1024), // 8 GB
        };
        let display = format!("{}", info);
        assert!(display.contains("CPU"));
        assert!(display.contains("Test CPU"));
        assert!(display.contains("16 GB"));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_available() {
        // Just test that the function doesn't panic
        let _ = is_metal_available();
    }

    /// The `metal` feature can be enabled on a target that cannot compile
    /// candle's Metal backend (`--all-features` on Linux/Windows does exactly
    /// that). Such a build must fail loudly rather than quietly hand back a
    /// CPU device labelled `Metal`.
    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_without_backend_fails_loudly() {
        if kizzasi_metal::BACKEND_COMPILED {
            return;
        }

        assert!(!is_metal_available());
        assert!(get_metal_devices().is_empty());
        assert!(matches!(get_best_device(), Device::Cpu));

        let err = DeviceConfig::new()
            .with_device_type(DeviceType::Metal)
            .create_device()
            .expect_err("a build without the Metal backend must not create a Metal device");
        assert!(matches!(err, CoreError::DeviceError(_)));
        assert!(
            err.to_string().contains("Apple"),
            "the error must name the cause, got: {err}"
        );
    }
}
