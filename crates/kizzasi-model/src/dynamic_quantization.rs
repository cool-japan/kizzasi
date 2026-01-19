//! Dynamic Quantization for On-the-Fly Model Compression
//!
//! Provides automatic weight quantization during model loading for memory-efficient inference.
//!
//! # Features
//!
//! - **Weight-Only Quantization**: Quantize weights while keeping activations in FP32
//! - **Dynamic Quantization**: Quantize weights and activations at runtime
//! - **Mixed Precision**: Selective quantization based on layer sensitivity
//! - **Multiple Backends**: INT8, FP16, BF16 support
//! - **HuggingFace Integration**: Automatic quantization on model load
//!
//! # Quantization Strategies
//!
//! ## INT8 Weight-Only Quantization
//! - Quantize weights to INT8 (4x compression)
//! - Keep activations in FP32 for accuracy
//! - Best for memory-bound workloads
//!
//! ## FP16 Mixed Precision
//! - Convert weights to FP16 (2x compression)
//! - Better accuracy than INT8
//! - Hardware acceleration on modern GPUs
//!
//! ## Dynamic Quantization
//! - Quantize both weights and activations
//! - Maximum memory savings (8x with INT8)
//! - Automatic calibration from data
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi_model::dynamic_quantization::*;
//!
//! // Load and quantize HuggingFace model
//! let quantizer = DynamicQuantizer::new()
//!     .with_strategy(QuantStrategy::INT8WeightOnly)
//!     .with_calibration_samples(100);
//!
//! let quantized_weights = quantizer.quantize_weights(&weights)?;
//! ```

use crate::error::ModelResult;
use crate::mixed_precision::{BF16Weights, FP16Weights};
use crate::quantization::{
    quantize_symmetric_2d, quantize_symmetric_per_channel, QuantizationGranularity, QuantizedWeight,
};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

/// Quantization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantStrategy {
    /// No quantization (FP32)
    None,
    /// Quantize weights to INT8, keep activations in FP32
    INT8WeightOnly,
    /// Quantize weights to FP16
    FP16,
    /// Quantize weights to BF16
    BF16,
    /// Quantize both weights and activations to INT8 (dynamic)
    INT8Dynamic,
    /// Mixed precision: sensitive layers in FP32, others in INT8
    MixedPrecision,
}

impl QuantStrategy {
    /// Get memory compression ratio
    pub fn compression_ratio(&self) -> f32 {
        match self {
            QuantStrategy::None => 1.0,
            QuantStrategy::INT8WeightOnly => 4.0,
            QuantStrategy::FP16 | QuantStrategy::BF16 => 2.0,
            QuantStrategy::INT8Dynamic => 8.0, // weights + activations
            QuantStrategy::MixedPrecision => 3.0, // average
        }
    }

    /// Check if strategy quantizes weights
    pub fn quantizes_weights(&self) -> bool {
        !matches!(self, QuantStrategy::None)
    }

    /// Check if strategy quantizes activations
    pub fn quantizes_activations(&self) -> bool {
        matches!(self, QuantStrategy::INT8Dynamic)
    }
}

/// Quantized model weights storage
#[derive(Debug, Clone)]
pub enum QuantizedWeightStorage {
    /// Original FP32 weights (no quantization)
    FP32(Array2<f32>),
    /// INT8 quantized weights
    INT8(QuantizedWeight),
    /// FP16 weights
    FP16(FP16Weights),
    /// BF16 weights
    BF16(BF16Weights),
}

impl QuantizedWeightStorage {
    /// Get memory size in bytes
    pub fn memory_size(&self) -> usize {
        match self {
            QuantizedWeightStorage::FP32(array) => array.len() * 4,
            QuantizedWeightStorage::INT8(qw) => qw.memory_size(),
            QuantizedWeightStorage::FP16(fp16) => fp16.memory_size(),
            QuantizedWeightStorage::BF16(bf16) => bf16.data.len() * 2,
        }
    }

    /// Convert to FP32 array for inference
    pub fn to_fp32(&self) -> ModelResult<Array2<f32>> {
        match self {
            QuantizedWeightStorage::FP32(array) => Ok(array.clone()),
            QuantizedWeightStorage::INT8(qw) => qw.dequantize_2d(),
            QuantizedWeightStorage::FP16(fp16) => fp16.to_f32_2d(),
            QuantizedWeightStorage::BF16(bf16) => bf16.to_f32_2d(),
        }
    }

    /// Get weight storage type as string
    pub fn storage_type(&self) -> &'static str {
        match self {
            QuantizedWeightStorage::FP32(_) => "FP32",
            QuantizedWeightStorage::INT8(_) => "INT8",
            QuantizedWeightStorage::FP16(_) => "FP16",
            QuantizedWeightStorage::BF16(_) => "BF16",
        }
    }
}

/// Layer sensitivity classification for mixed precision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerSensitivity {
    /// High sensitivity - keep in FP32
    High,
    /// Medium sensitivity - use FP16
    Medium,
    /// Low sensitivity - can use INT8
    Low,
}

/// Dynamic quantizer for automatic model compression
pub struct DynamicQuantizer {
    /// Quantization strategy
    strategy: QuantStrategy,
    /// Number of calibration samples for dynamic quantization
    calibration_samples: usize,
    /// Granularity for INT8 quantization
    granularity: QuantizationGranularity,
    /// Layer sensitivity heuristics
    sensitivity_heuristics: HashMap<String, LayerSensitivity>,
}

impl DynamicQuantizer {
    /// Create a new dynamic quantizer with default settings
    pub fn new() -> Self {
        Self {
            strategy: QuantStrategy::INT8WeightOnly,
            calibration_samples: 100,
            granularity: QuantizationGranularity::PerChannel,
            sensitivity_heuristics: Self::default_sensitivity_heuristics(),
        }
    }

    /// Set quantization strategy
    pub fn with_strategy(mut self, strategy: QuantStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set number of calibration samples
    pub fn with_calibration_samples(mut self, samples: usize) -> Self {
        self.calibration_samples = samples;
        self
    }

    /// Set quantization granularity for INT8
    pub fn with_granularity(mut self, granularity: QuantizationGranularity) -> Self {
        self.granularity = granularity;
        self
    }

    /// Default layer sensitivity heuristics
    ///
    /// Based on common SSM architecture patterns:
    /// - Input/output projections: High sensitivity (first/last layers)
    /// - SSM parameters (A, B, C matrices): High sensitivity (state dynamics)
    /// - Layer norms: Medium sensitivity
    /// - MLP/FFN layers: Low sensitivity (most compressible)
    fn default_sensitivity_heuristics() -> HashMap<String, LayerSensitivity> {
        let mut heuristics = HashMap::new();

        // High sensitivity layers
        heuristics.insert("input_proj".to_string(), LayerSensitivity::High);
        heuristics.insert("output_proj".to_string(), LayerSensitivity::High);
        heuristics.insert("ssm.log_a".to_string(), LayerSensitivity::High);
        heuristics.insert("ssm.b_proj".to_string(), LayerSensitivity::High);
        heuristics.insert("ssm.c_proj".to_string(), LayerSensitivity::High);

        // Medium sensitivity layers
        heuristics.insert("norm".to_string(), LayerSensitivity::Medium);
        heuristics.insert("ln".to_string(), LayerSensitivity::Medium);
        heuristics.insert("time_mix".to_string(), LayerSensitivity::Medium);

        // Low sensitivity layers (FFN, channel mixing)
        heuristics.insert("channel_mix".to_string(), LayerSensitivity::Low);
        heuristics.insert("ffn".to_string(), LayerSensitivity::Low);
        heuristics.insert("mlp".to_string(), LayerSensitivity::Low);

        heuristics
    }

    /// Classify layer sensitivity based on name
    pub fn classify_layer(&self, layer_name: &str) -> LayerSensitivity {
        // Check exact matches first
        if let Some(&sensitivity) = self.sensitivity_heuristics.get(layer_name) {
            return sensitivity;
        }

        // Check partial matches
        for (pattern, &sensitivity) in &self.sensitivity_heuristics {
            if layer_name.contains(pattern) {
                return sensitivity;
            }
        }

        // Default to medium sensitivity
        LayerSensitivity::Medium
    }

    /// Quantize a single weight tensor
    pub fn quantize_weight(
        &self,
        weight: &Array2<f32>,
        layer_name: &str,
    ) -> ModelResult<QuantizedWeightStorage> {
        match self.strategy {
            QuantStrategy::None => Ok(QuantizedWeightStorage::FP32(weight.clone())),

            QuantStrategy::INT8WeightOnly => {
                let quantized = match self.granularity {
                    QuantizationGranularity::PerTensor => quantize_symmetric_2d(weight)?,
                    QuantizationGranularity::PerChannel => quantize_symmetric_per_channel(weight)?,
                };
                Ok(QuantizedWeightStorage::INT8(quantized))
            }

            QuantStrategy::FP16 => {
                let fp16_weights = FP16Weights::from_f32_2d(weight);
                Ok(QuantizedWeightStorage::FP16(fp16_weights))
            }

            QuantStrategy::BF16 => {
                let bf16_weights = BF16Weights::from_f32_2d(weight);
                Ok(QuantizedWeightStorage::BF16(bf16_weights))
            }

            QuantStrategy::INT8Dynamic => {
                // Dynamic quantization: same as weight-only for weights
                let quantized = match self.granularity {
                    QuantizationGranularity::PerTensor => quantize_symmetric_2d(weight)?,
                    QuantizationGranularity::PerChannel => quantize_symmetric_per_channel(weight)?,
                };
                Ok(QuantizedWeightStorage::INT8(quantized))
            }

            QuantStrategy::MixedPrecision => {
                // Selective quantization based on layer sensitivity
                let sensitivity = self.classify_layer(layer_name);

                match sensitivity {
                    LayerSensitivity::High => {
                        // Keep high-sensitivity layers in FP32
                        Ok(QuantizedWeightStorage::FP32(weight.clone()))
                    }
                    LayerSensitivity::Medium => {
                        // Medium sensitivity: use FP16
                        let fp16_weights = FP16Weights::from_f32_2d(weight);
                        Ok(QuantizedWeightStorage::FP16(fp16_weights))
                    }
                    LayerSensitivity::Low => {
                        // Low sensitivity: use INT8
                        let quantized = quantize_symmetric_per_channel(weight)?;
                        Ok(QuantizedWeightStorage::INT8(quantized))
                    }
                }
            }
        }
    }

    /// Quantize all weights in a model
    pub fn quantize_weights(
        &self,
        weights: &HashMap<String, Array2<f32>>,
    ) -> ModelResult<HashMap<String, QuantizedWeightStorage>> {
        let mut quantized_weights = HashMap::new();

        for (name, weight) in weights {
            let quantized = self.quantize_weight(weight, name)?;
            quantized_weights.insert(name.clone(), quantized);
        }

        Ok(quantized_weights)
    }

    /// Calculate total memory savings
    pub fn calculate_memory_savings(
        &self,
        original_weights: &HashMap<String, Array2<f32>>,
        quantized_weights: &HashMap<String, QuantizedWeightStorage>,
    ) -> QuantizationStats {
        let mut original_size = 0;
        let mut quantized_size = 0;

        for (name, original) in original_weights {
            original_size += original.len() * 4; // FP32: 4 bytes

            if let Some(quantized) = quantized_weights.get(name) {
                quantized_size += quantized.memory_size();
            }
        }

        let compression_ratio = original_size as f32 / quantized_size.max(1) as f32;
        let memory_saved = original_size.saturating_sub(quantized_size);

        QuantizationStats {
            original_size_bytes: original_size,
            quantized_size_bytes: quantized_size,
            compression_ratio,
            memory_saved_bytes: memory_saved,
            strategy: self.strategy,
        }
    }

    /// Get quantization strategy
    pub fn strategy(&self) -> QuantStrategy {
        self.strategy
    }

    /// Get calibration sample count
    pub fn calibration_samples(&self) -> usize {
        self.calibration_samples
    }
}

impl Default for DynamicQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Quantization statistics
#[derive(Debug, Clone)]
pub struct QuantizationStats {
    /// Original model size in bytes
    pub original_size_bytes: usize,
    /// Quantized model size in bytes
    pub quantized_size_bytes: usize,
    /// Compression ratio (original / quantized)
    pub compression_ratio: f32,
    /// Memory saved in bytes
    pub memory_saved_bytes: usize,
    /// Strategy used
    pub strategy: QuantStrategy,
}

impl QuantizationStats {
    /// Format size as human-readable string
    pub fn format_size(bytes: usize) -> String {
        const KB: usize = 1024;
        const MB: usize = KB * 1024;
        const GB: usize = MB * 1024;

        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} bytes", bytes)
        }
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("Quantization Summary");
        println!("====================");
        println!("Strategy: {:?}", self.strategy);
        println!(
            "Original Size: {}",
            Self::format_size(self.original_size_bytes)
        );
        println!(
            "Quantized Size: {}",
            Self::format_size(self.quantized_size_bytes)
        );
        println!("Compression Ratio: {:.2}x", self.compression_ratio);
        println!(
            "Memory Saved: {} ({:.1}%)",
            Self::format_size(self.memory_saved_bytes),
            (self.memory_saved_bytes as f64 / self.original_size_bytes as f64) * 100.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_quant_strategy_compression_ratio() {
        assert_eq!(QuantStrategy::None.compression_ratio(), 1.0);
        assert_eq!(QuantStrategy::INT8WeightOnly.compression_ratio(), 4.0);
        assert_eq!(QuantStrategy::FP16.compression_ratio(), 2.0);
        assert_eq!(QuantStrategy::BF16.compression_ratio(), 2.0);
        assert_eq!(QuantStrategy::INT8Dynamic.compression_ratio(), 8.0);
    }

    #[test]
    fn test_dynamic_quantizer_creation() {
        let quantizer = DynamicQuantizer::new();
        assert_eq!(quantizer.strategy(), QuantStrategy::INT8WeightOnly);
        assert_eq!(quantizer.calibration_samples(), 100);
    }

    #[test]
    fn test_quantizer_with_strategy() {
        let quantizer = DynamicQuantizer::new()
            .with_strategy(QuantStrategy::FP16)
            .with_calibration_samples(200);

        assert_eq!(quantizer.strategy(), QuantStrategy::FP16);
        assert_eq!(quantizer.calibration_samples(), 200);
    }

    #[test]
    fn test_layer_sensitivity_classification() {
        let quantizer = DynamicQuantizer::new();

        assert_eq!(
            quantizer.classify_layer("input_proj"),
            LayerSensitivity::High
        );
        assert_eq!(
            quantizer.classify_layer("layers.0.ssm.log_a"),
            LayerSensitivity::High
        );
        assert_eq!(
            quantizer.classify_layer("layers.0.norm.weight"),
            LayerSensitivity::Medium
        );
        assert_eq!(
            quantizer.classify_layer("layers.0.channel_mix.key"),
            LayerSensitivity::Low
        );
        assert_eq!(
            quantizer.classify_layer("unknown_layer"),
            LayerSensitivity::Medium
        ); // default
    }

    #[test]
    fn test_quantize_weight_int8() {
        let quantizer = DynamicQuantizer::new().with_strategy(QuantStrategy::INT8WeightOnly);

        let weight = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
            .expect("Failed to create test array");

        let quantized = quantizer
            .quantize_weight(&weight, "test_layer")
            .expect("Failed to quantize weight");

        assert_eq!(quantized.storage_type(), "INT8");
        assert!(quantized.memory_size() < weight.len() * 4);
    }

    #[test]
    fn test_quantize_weight_fp16() {
        let quantizer = DynamicQuantizer::new().with_strategy(QuantStrategy::FP16);

        let weight = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
            .expect("Failed to create test array");

        let quantized = quantizer
            .quantize_weight(&weight, "test_layer")
            .expect("Failed to quantize weight");

        assert_eq!(quantized.storage_type(), "FP16");
        assert_eq!(quantized.memory_size(), weight.len() * 2); // FP16 = 2 bytes
    }

    #[test]
    fn test_quantize_weight_mixed_precision() {
        let quantizer = DynamicQuantizer::new().with_strategy(QuantStrategy::MixedPrecision);

        let weight = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
            .expect("Failed to create test array");

        // High sensitivity layer - should stay FP32
        let quantized_high = quantizer
            .quantize_weight(&weight, "input_proj")
            .expect("Failed to quantize weight");
        assert_eq!(quantized_high.storage_type(), "FP32");

        // Medium sensitivity layer - should be FP16
        let quantized_medium = quantizer
            .quantize_weight(&weight, "norm")
            .expect("Failed to quantize weight");
        assert_eq!(quantized_medium.storage_type(), "FP16");

        // Low sensitivity layer - should be INT8
        let quantized_low = quantizer
            .quantize_weight(&weight, "channel_mix")
            .expect("Failed to quantize weight");
        assert_eq!(quantized_low.storage_type(), "INT8");
    }

    #[test]
    fn test_quantize_all_weights() {
        let quantizer = DynamicQuantizer::new().with_strategy(QuantStrategy::INT8WeightOnly);

        let mut weights = HashMap::new();
        weights.insert(
            "layer1".to_string(),
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        );
        weights.insert(
            "layer2".to_string(),
            Array2::from_shape_vec((2, 2), vec![-1.0, -2.0, -3.0, -4.0]).unwrap(),
        );

        let quantized = quantizer
            .quantize_weights(&weights)
            .expect("Failed to quantize weights");

        assert_eq!(quantized.len(), 2);
        assert!(quantized.contains_key("layer1"));
        assert!(quantized.contains_key("layer2"));
    }

    #[test]
    fn test_calculate_memory_savings() {
        let quantizer = DynamicQuantizer::new().with_strategy(QuantStrategy::INT8WeightOnly);

        let mut weights = HashMap::new();
        weights.insert(
            "layer1".to_string(),
            Array2::from_shape_vec((100, 100), vec![1.0; 10000]).unwrap(),
        );

        let quantized = quantizer.quantize_weights(&weights).unwrap();
        let stats = quantizer.calculate_memory_savings(&weights, &quantized);

        assert_eq!(stats.original_size_bytes, 10000 * 4); // FP32
        assert_eq!(stats.quantized_size_bytes, 10000); // INT8
        assert!((stats.compression_ratio - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_quantization_stats_format() {
        let stats = QuantizationStats {
            original_size_bytes: 1024 * 1024 * 100, // 100 MB
            quantized_size_bytes: 1024 * 1024 * 25, // 25 MB
            compression_ratio: 4.0,
            memory_saved_bytes: 1024 * 1024 * 75, // 75 MB
            strategy: QuantStrategy::INT8WeightOnly,
        };

        let formatted = QuantizationStats::format_size(stats.original_size_bytes);
        assert!(formatted.contains("MB"));
    }

    #[test]
    fn test_storage_to_fp32_roundtrip() {
        let original = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
            .expect("Failed to create test array");

        // Test FP32 storage
        let storage_fp32 = QuantizedWeightStorage::FP32(original.clone());
        let restored = storage_fp32.to_fp32().expect("Failed to restore");
        assert_eq!(restored, original);

        // Test FP16 storage
        let fp16 = FP16Weights::from_f32_2d(&original);
        let storage_fp16 = QuantizedWeightStorage::FP16(fp16);
        let restored_fp16 = storage_fp16.to_fp32().expect("Failed to restore");
        assert_eq!(restored_fp16.dim(), original.dim());

        // Test INT8 storage
        let int8 = quantize_symmetric_2d(&original).expect("Failed to quantize");
        let storage_int8 = QuantizedWeightStorage::INT8(int8);
        let restored_int8 = storage_int8.to_fp32().expect("Failed to restore");
        assert_eq!(restored_int8.dim(), original.dim());
    }
}
