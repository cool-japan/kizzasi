//! Multi-modal inference pipeline support
//!
//! Provides infrastructure for handling multiple input modalities and fusing them
//! for unified inference. Supports various fusion strategies and modality-specific
//! preprocessing.
//!
//! # Supported Modalities
//!
//! - Audio: Time-series audio signals
//! - Video: Frame-based visual data
//! - Sensor: Generic sensor readings (IMU, temperature, etc.)
//! - Text: Embedded text representations
//!
//! # Fusion Strategies
//!
//! - Early fusion: Concatenate modalities before model
//! - Late fusion: Process separately, combine outputs
//! - Cross-attention: Attend across modalities
//! - Hierarchical: Multi-level fusion
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi_inference::multimodal::{MultiModalPipeline, ModalityType, FusionStrategy};
//!
//! let mut pipeline = MultiModalPipeline::builder()
//!     .add_modality(ModalityType::Audio, audio_processor)
//!     .add_modality(ModalityType::Video, video_processor)
//!     .fusion_strategy(FusionStrategy::EarlyFusion)
//!     .build()?;
//!
//! let audio_input = Array1::from_vec(vec![0.1, 0.2, 0.3]);
//! let video_input = Array1::from_vec(vec![0.4, 0.5, 0.6]);
//!
//! let output = pipeline.forward(&[
//!     (ModalityType::Audio, audio_input),
//!     (ModalityType::Video, video_input),
//! ])?;
//! ```

use crate::engine::{EngineConfig, InferenceEngine};
use crate::error::{InferenceError, InferenceResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Supported modality types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModalityType {
    /// Audio signals (time-series)
    Audio,
    /// Video frames (image sequences)
    Video,
    /// Sensor data (IMU, temperature, pressure, etc.)
    Sensor,
    /// Text embeddings
    Text,
    /// Custom modality
    Custom(&'static str),
}

impl ModalityType {
    /// Get the modality name as a string
    pub fn name(&self) -> &str {
        match self {
            ModalityType::Audio => "audio",
            ModalityType::Video => "video",
            ModalityType::Sensor => "sensor",
            ModalityType::Text => "text",
            ModalityType::Custom(name) => name,
        }
    }
}

/// Fusion strategy for combining multiple modalities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FusionStrategy {
    /// Concatenate all modalities before processing
    #[default]
    EarlyFusion,
    /// Process each modality separately, then combine outputs
    LateFusion,
    /// Weighted average of modality outputs
    WeightedFusion,
    /// Maximum pooling across modalities
    MaxPooling,
    /// Cross-attention between modalities
    CrossAttention,
    /// Hierarchical multi-level fusion
    Hierarchical,
}

/// Preprocessor for a specific modality
pub type ModalityPreprocessor =
    Arc<dyn Fn(&Array1<f32>) -> InferenceResult<Array1<f32>> + Send + Sync>;

/// Configuration for a single modality
#[derive(Clone)]
pub struct ModalityConfig {
    /// Type of modality
    pub modality_type: ModalityType,
    /// Expected input dimension
    pub input_dim: usize,
    /// Optional preprocessor
    pub preprocessor: Option<ModalityPreprocessor>,
    /// Weight for fusion (used in weighted fusion)
    pub fusion_weight: f32,
}

impl ModalityConfig {
    /// Create a new modality configuration
    pub fn new(modality_type: ModalityType, input_dim: usize) -> Self {
        Self {
            modality_type,
            input_dim,
            preprocessor: None,
            fusion_weight: 1.0,
        }
    }

    /// Set the preprocessor
    pub fn preprocessor(mut self, preprocessor: ModalityPreprocessor) -> Self {
        self.preprocessor = Some(preprocessor);
        self
    }

    /// Set the fusion weight
    pub fn fusion_weight(mut self, weight: f32) -> Self {
        self.fusion_weight = weight;
        self
    }
}

/// Multi-modal inference pipeline
pub struct MultiModalPipeline {
    /// Base inference engine
    engine: InferenceEngine,
    /// Modality configurations
    modalities: HashMap<ModalityType, ModalityConfig>,
    /// Fusion strategy
    fusion_strategy: FusionStrategy,
    /// Total expected input dimension (sum of all modality dims)
    #[allow(dead_code)]
    total_input_dim: usize,
}

impl MultiModalPipeline {
    /// Create a new builder
    pub fn builder() -> MultiModalPipelineBuilder {
        MultiModalPipelineBuilder::new()
    }

    /// Forward pass with multi-modal inputs
    ///
    /// # Arguments
    ///
    /// * `inputs` - Slice of (modality_type, input_array) pairs
    ///
    /// # Returns
    ///
    /// Fused output array
    pub fn forward(
        &mut self,
        inputs: &[(ModalityType, Array1<f32>)],
    ) -> InferenceResult<Array1<f32>> {
        // Validate inputs
        for (modality, input) in inputs {
            let config = self.modalities.get(modality).ok_or_else(|| {
                InferenceError::PipelineConfig(format!("Unknown modality: {:?}", modality))
            })?;

            if input.len() != config.input_dim {
                return Err(InferenceError::DimensionMismatch {
                    expected: config.input_dim,
                    got: input.len(),
                });
            }
        }

        // Preprocess each modality
        let mut preprocessed: HashMap<ModalityType, Array1<f32>> = HashMap::new();
        for (modality, input) in inputs {
            let config = &self.modalities[modality];
            let processed = if let Some(ref preprocessor) = config.preprocessor {
                preprocessor(input)?
            } else {
                input.clone()
            };
            preprocessed.insert(*modality, processed);
        }

        // Apply fusion strategy
        let fused = self.fuse(&preprocessed)?;

        // Run inference
        self.engine.step(&fused)
    }

    /// Apply fusion strategy to combine modalities
    fn fuse(
        &mut self,
        inputs: &HashMap<ModalityType, Array1<f32>>,
    ) -> InferenceResult<Array1<f32>> {
        match self.fusion_strategy {
            FusionStrategy::EarlyFusion => self.early_fusion(inputs),
            FusionStrategy::LateFusion => self.late_fusion(inputs),
            FusionStrategy::WeightedFusion => self.weighted_fusion(inputs),
            FusionStrategy::MaxPooling => self.max_pooling_fusion(inputs),
            FusionStrategy::CrossAttention => self.cross_attention_fusion(inputs),
            FusionStrategy::Hierarchical => self.hierarchical_fusion(inputs),
        }
    }

    /// Modalities present in `inputs`, ordered deterministically by
    /// [`ModalityType::name`].
    ///
    /// `HashMap` iteration order is randomised per process, so every fusion
    /// strategy that concatenates or aligns per-modality segments must
    /// iterate through this helper rather than the map directly —
    /// otherwise two fusion strategies invoked in the same call (as
    /// [`MultiModalPipeline::hierarchical_fusion`] does with
    /// [`MultiModalPipeline::early_fusion`] and
    /// [`MultiModalPipeline::weighted_fusion`]) can lay modalities out in
    /// different orders and silently average one modality's samples with
    /// another's.
    fn sorted_modality_keys(inputs: &HashMap<ModalityType, Array1<f32>>) -> Vec<ModalityType> {
        let mut keys: Vec<ModalityType> = inputs.keys().copied().collect();
        // `sort_by` rather than `sort_by_key`: `ModalityType::name`'s elided
        // signature ties its `&str` return to `&self`'s borrow, which
        // `sort_by_key` cannot thread through its `K: Ord` extraction.
        keys.sort_by(|a, b| a.name().cmp(b.name()));
        keys
    }

    /// Early fusion: concatenate all modalities
    fn early_fusion(
        &self,
        inputs: &HashMap<ModalityType, Array1<f32>>,
    ) -> InferenceResult<Array1<f32>> {
        let mut result = Vec::new();

        for modality in Self::sorted_modality_keys(inputs) {
            let input = &inputs[&modality];
            let slice = input.as_slice().ok_or_else(|| {
                InferenceError::ForwardError(
                    "Array data not contiguous in early fusion".to_string(),
                )
            })?;
            result.extend_from_slice(slice);
        }

        Ok(Array1::from_vec(result))
    }

    /// Late fusion: average modalities element-wise, then concatenate
    ///
    /// Note: With a single engine, true late fusion (separate processing per modality)
    /// isn't possible. This implements a simplified version that averages modalities
    /// of the same dimension before concatenation.
    fn late_fusion(
        &mut self,
        inputs: &HashMap<ModalityType, Array1<f32>>,
    ) -> InferenceResult<Array1<f32>> {
        if inputs.is_empty() {
            return Err(InferenceError::PipelineConfig(
                "No modalities to fuse".into(),
            ));
        }

        // Group modalities by dimension, inserting in deterministic
        // (name-sorted) order so groups sharing a dimension have a stable
        // internal layout too.
        let mut by_dim: std::collections::HashMap<usize, Vec<Array1<f32>>> =
            std::collections::HashMap::new();
        for modality in Self::sorted_modality_keys(inputs) {
            let input = &inputs[&modality];
            by_dim.entry(input.len()).or_default().push(input.clone());
        }

        // Average within each dimension group, then concatenate
        let mut result = Vec::new();
        let mut dims: Vec<_> = by_dim.keys().cloned().collect();
        dims.sort();

        for dim in dims {
            let arrays = &by_dim[&dim];
            let mut averaged = Array1::zeros(dim);
            for arr in arrays {
                averaged += arr;
            }
            averaged /= arrays.len() as f32;
            let slice = averaged.as_slice().ok_or_else(|| {
                InferenceError::ForwardError("Array data not contiguous in late fusion".to_string())
            })?;
            result.extend_from_slice(slice);
        }

        Ok(Array1::from_vec(result))
    }

    /// Weighted fusion: weighted average based on modality weights
    fn weighted_fusion(
        &self,
        inputs: &HashMap<ModalityType, Array1<f32>>,
    ) -> InferenceResult<Array1<f32>> {
        let mut result = Vec::new();
        let mut total_weight = 0.0;

        // Weighted concatenation, in the same deterministic (name-sorted)
        // order as every other fusion strategy — `HashMap` iteration order
        // is randomised per process, so iterating `inputs` directly used to
        // lay this strategy's output out differently from `early_fusion`'s
        // on every other run, silently mixing modalities together wherever
        // `hierarchical_fusion` averages the two element-wise.
        for modality in Self::sorted_modality_keys(inputs) {
            let input = &inputs[&modality];
            let config = &self.modalities[&modality];
            let weight = config.fusion_weight;
            total_weight += weight;

            let weighted = input.mapv(|x| x * weight);
            let slice = weighted.as_slice().ok_or_else(|| {
                InferenceError::ForwardError(
                    "Array data not contiguous in weighted fusion".to_string(),
                )
            })?;
            result.extend_from_slice(slice);
        }

        // Normalize by total weight
        let normalized: Vec<f32> = result.iter().map(|x| x / total_weight).collect();
        Ok(Array1::from_vec(normalized))
    }

    /// Max pooling fusion: element-wise max across modalities of same dimension, then concatenate
    fn max_pooling_fusion(
        &self,
        inputs: &HashMap<ModalityType, Array1<f32>>,
    ) -> InferenceResult<Array1<f32>> {
        if inputs.is_empty() {
            return Err(InferenceError::PipelineConfig(
                "No modalities to fuse".into(),
            ));
        }

        // Group modalities by dimension, in deterministic (name-sorted) order.
        let mut by_dim: std::collections::HashMap<usize, Vec<Array1<f32>>> =
            std::collections::HashMap::new();
        for modality in Self::sorted_modality_keys(inputs) {
            let input = &inputs[&modality];
            by_dim.entry(input.len()).or_default().push(input.clone());
        }

        // Max pool within each dimension group, then concatenate
        let mut result = Vec::new();
        let mut dims: Vec<_> = by_dim.keys().cloned().collect();
        dims.sort();

        for dim in dims {
            let arrays = &by_dim[&dim];
            if arrays.len() == 1 {
                let slice = arrays[0].as_slice().ok_or_else(|| {
                    InferenceError::ForwardError(
                        "Array data not contiguous in max pooling".to_string(),
                    )
                })?;
                result.extend_from_slice(slice);
            } else {
                // Stack and max pool
                let nrows = arrays.len();
                let ncols = dim;
                let mut stacked = Array2::zeros((nrows, ncols));
                for (i, arr) in arrays.iter().enumerate() {
                    for (j, &val) in arr.iter().enumerate() {
                        stacked[[i, j]] = val;
                    }
                }
                let pooled = stacked.map_axis(Axis(0), |col| {
                    col.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                });
                let slice = pooled.as_slice().ok_or_else(|| {
                    InferenceError::ForwardError(
                        "Array data not contiguous in max pooling result".to_string(),
                    )
                })?;
                result.extend_from_slice(slice);
            }
        }

        Ok(Array1::from_vec(result))
    }

    /// Cross-attention fusion: attention-weighted combination then concatenation
    fn cross_attention_fusion(
        &self,
        inputs: &HashMap<ModalityType, Array1<f32>>,
    ) -> InferenceResult<Array1<f32>> {
        if inputs.is_empty() {
            return Err(InferenceError::PipelineConfig(
                "No modalities to fuse".into(),
            ));
        }

        if inputs.len() == 1 {
            let single_input = inputs.values().next().ok_or_else(|| {
                InferenceError::ForwardError("No input found in hierarchical fusion".to_string())
            })?;
            return Ok(single_input.clone());
        }

        // Group modalities by dimension, in deterministic (name-sorted) order.
        let mut by_dim: std::collections::HashMap<usize, Vec<Array1<f32>>> =
            std::collections::HashMap::new();
        for modality in Self::sorted_modality_keys(inputs) {
            let input = &inputs[&modality];
            by_dim.entry(input.len()).or_default().push(input.clone());
        }

        // Apply attention within each dimension group, then concatenate
        let mut result = Vec::new();
        let mut dims: Vec<_> = by_dim.keys().cloned().collect();
        dims.sort();

        for dim in dims {
            let modalities = &by_dim[&dim];
            let n = modalities.len();

            if n == 1 {
                let slice = modalities[0].as_slice().ok_or_else(|| {
                    InferenceError::ForwardError(
                        "Array data not contiguous in hierarchical fusion".to_string(),
                    )
                })?;
                result.extend_from_slice(slice);
            } else {
                // Compute pairwise attention scores (dot products)
                let mut attention_weights = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        if i != j {
                            let dot_product: f32 = modalities[i]
                                .iter()
                                .zip(modalities[j].iter())
                                .map(|(a, b)| a * b)
                                .sum();
                            attention_weights[i] += dot_product.abs();
                        }
                    }
                }

                // Normalize attention weights
                let total: f32 = attention_weights.iter().sum();
                if total > 0.0 {
                    for weight in &mut attention_weights {
                        *weight /= total;
                    }
                } else {
                    // Uniform weights if no attention signal
                    let uniform = 1.0 / n as f32;
                    attention_weights.fill(uniform);
                }

                // Weighted sum
                let mut weighted_result = Array1::zeros(dim);
                for (i, modality) in modalities.iter().enumerate() {
                    weighted_result += &(modality * attention_weights[i]);
                }
                let slice = weighted_result.as_slice().ok_or_else(|| {
                    InferenceError::ForwardError(
                        "Array data not contiguous in cross-attention result".to_string(),
                    )
                })?;
                result.extend_from_slice(slice);
            }
        }

        Ok(Array1::from_vec(result))
    }

    /// Hierarchical fusion: multi-level combination
    ///
    /// Combines early fusion (direct concatenation) and weighted fusion
    /// by blending their results element-wise.
    fn hierarchical_fusion(
        &mut self,
        inputs: &HashMap<ModalityType, Array1<f32>>,
    ) -> InferenceResult<Array1<f32>> {
        // First level: early fusion (direct concatenation)
        let early = self.early_fusion(inputs)?;

        // Second level: weighted fusion (using modality weights)
        let weighted = self.weighted_fusion(inputs)?;

        // Both should have the same dimension
        if early.len() != weighted.len() {
            return Err(InferenceError::PipelineConfig(format!(
                "Fusion dimension mismatch: early={}, weighted={}",
                early.len(),
                weighted.len()
            )));
        }

        // Blend both strategies (element-wise average)
        let result = (early + weighted) / 2.0;
        Ok(result)
    }

    /// Reset the pipeline state
    pub fn reset(&mut self) {
        self.engine.reset();
    }

    /// Get the fusion strategy
    pub fn fusion_strategy(&self) -> FusionStrategy {
        self.fusion_strategy
    }

    /// Get modality configurations
    pub fn modalities(&self) -> &HashMap<ModalityType, ModalityConfig> {
        &self.modalities
    }

    /// Get the underlying engine
    pub fn engine(&self) -> &InferenceEngine {
        &self.engine
    }

    /// Get mutable access to the engine
    pub fn engine_mut(&mut self) -> &mut InferenceEngine {
        &mut self.engine
    }
}

/// Builder for multi-modal pipelines
pub struct MultiModalPipelineBuilder {
    engine_config: Option<EngineConfig>,
    modalities: HashMap<ModalityType, ModalityConfig>,
    fusion_strategy: FusionStrategy,
}

impl MultiModalPipelineBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            engine_config: None,
            modalities: HashMap::new(),
            fusion_strategy: FusionStrategy::default(),
        }
    }

    /// Set the engine configuration
    pub fn engine_config(mut self, config: EngineConfig) -> Self {
        self.engine_config = Some(config);
        self
    }

    /// Add a modality configuration
    pub fn add_modality(mut self, config: ModalityConfig) -> Self {
        self.modalities.insert(config.modality_type, config);
        self
    }

    /// Add a modality with type and dimension
    pub fn modality(mut self, modality_type: ModalityType, input_dim: usize) -> Self {
        let config = ModalityConfig::new(modality_type, input_dim);
        self.modalities.insert(modality_type, config);
        self
    }

    /// Set the fusion strategy
    pub fn fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.fusion_strategy = strategy;
        self
    }

    /// Build the pipeline
    pub fn build(self) -> InferenceResult<MultiModalPipeline> {
        if self.modalities.is_empty() {
            return Err(InferenceError::PipelineConfig(
                "At least one modality must be configured".into(),
            ));
        }

        // Calculate total input dimension
        let total_input_dim: usize = self.modalities.values().map(|c| c.input_dim).sum();

        let engine_config = self
            .engine_config
            .ok_or_else(|| InferenceError::PipelineConfig("engine_config not set".into()))?;

        let engine = InferenceEngine::new(engine_config);

        Ok(MultiModalPipeline {
            engine,
            modalities: self.modalities,
            fusion_strategy: self.fusion_strategy,
            total_input_dim,
        })
    }
}

impl Default for MultiModalPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kizzasi_model::s4::{S4Config, S4D};

    fn create_test_model(input_dim: usize, _output_dim: usize) -> Box<S4D> {
        let config = S4Config::new()
            .input_dim(input_dim)
            .hidden_dim(32)
            .state_dim(16)
            .num_layers(1)
            .diagonal(true);
        Box::new(S4D::new(config).unwrap())
    }

    #[test]
    fn test_modality_type_name() {
        assert_eq!(ModalityType::Audio.name(), "audio");
        assert_eq!(ModalityType::Video.name(), "video");
        assert_eq!(ModalityType::Sensor.name(), "sensor");
        assert_eq!(ModalityType::Text.name(), "text");
        assert_eq!(ModalityType::Custom("xyz").name(), "xyz");
    }

    #[test]
    fn test_multimodal_builder() {
        let engine_config = EngineConfig::new(6, 10);
        let pipeline = MultiModalPipeline::builder()
            .engine_config(engine_config)
            .modality(ModalityType::Audio, 3)
            .modality(ModalityType::Video, 3)
            .fusion_strategy(FusionStrategy::EarlyFusion)
            .build();

        assert!(pipeline.is_ok());
        let p = pipeline.unwrap();
        assert_eq!(p.modalities().len(), 2);
        assert_eq!(p.total_input_dim, 6);
    }

    #[test]
    fn test_multimodal_no_modalities() {
        let engine_config = EngineConfig::new(3, 10);
        let result = MultiModalPipeline::builder()
            .engine_config(engine_config)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_early_fusion() {
        let engine_config = EngineConfig::new(6, 6);
        let mut pipeline = MultiModalPipeline::builder()
            .engine_config(engine_config)
            .modality(ModalityType::Audio, 3)
            .modality(ModalityType::Video, 3)
            .fusion_strategy(FusionStrategy::EarlyFusion)
            .build()
            .unwrap();

        pipeline.engine_mut().set_model(create_test_model(6, 6));

        let audio = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let video = Array1::from_vec(vec![0.4, 0.5, 0.6]);

        let result =
            pipeline.forward(&[(ModalityType::Audio, audio), (ModalityType::Video, video)]);

        result.unwrap(); // This will show the error
    }

    #[test]
    fn test_weighted_fusion() {
        let engine_config = EngineConfig::new(4, 4);

        let audio_config = ModalityConfig::new(ModalityType::Audio, 2).fusion_weight(2.0);
        let video_config = ModalityConfig::new(ModalityType::Video, 2).fusion_weight(1.0);

        let mut pipeline = MultiModalPipeline::builder()
            .engine_config(engine_config)
            .add_modality(audio_config)
            .add_modality(video_config)
            .fusion_strategy(FusionStrategy::WeightedFusion)
            .build()
            .unwrap();

        pipeline.engine_mut().set_model(create_test_model(4, 4));

        let audio = Array1::from_vec(vec![0.3, 0.6]);
        let video = Array1::from_vec(vec![0.1, 0.2]);

        let result =
            pipeline.forward(&[(ModalityType::Audio, audio), (ModalityType::Video, video)]);

        assert!(result.is_ok());
    }

    #[test]
    fn test_dimension_mismatch() {
        let engine_config = EngineConfig::new(6, 10);
        let mut pipeline = MultiModalPipeline::builder()
            .engine_config(engine_config)
            .modality(ModalityType::Audio, 3)
            .modality(ModalityType::Video, 3)
            .build()
            .unwrap();

        let audio = Array1::from_vec(vec![0.1, 0.2]); // Wrong dimension!
        let video = Array1::from_vec(vec![0.4, 0.5, 0.6]);

        let result =
            pipeline.forward(&[(ModalityType::Audio, audio), (ModalityType::Video, video)]);

        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_modality() {
        let engine_config = EngineConfig::new(3, 10);
        let mut pipeline = MultiModalPipeline::builder()
            .engine_config(engine_config)
            .modality(ModalityType::Audio, 3)
            .build()
            .unwrap();

        let video = Array1::from_vec(vec![0.4, 0.5, 0.6]);

        let result = pipeline.forward(&[(ModalityType::Video, video)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_max_pooling_fusion() {
        let engine_config = EngineConfig::new(3, 3);
        let mut pipeline = MultiModalPipeline::builder()
            .engine_config(engine_config)
            .modality(ModalityType::Audio, 3)
            .modality(ModalityType::Video, 3)
            .fusion_strategy(FusionStrategy::MaxPooling)
            .build()
            .unwrap();

        pipeline.engine_mut().set_model(create_test_model(3, 3));

        let audio = Array1::from_vec(vec![0.1, 0.9, 0.3]);
        let video = Array1::from_vec(vec![0.8, 0.2, 0.6]);

        let result =
            pipeline.forward(&[(ModalityType::Audio, audio), (ModalityType::Video, video)]);

        assert!(result.is_ok());
    }

    #[test]
    fn test_cross_attention_fusion() {
        let engine_config = EngineConfig::new(3, 3);
        let mut pipeline = MultiModalPipeline::builder()
            .engine_config(engine_config)
            .modality(ModalityType::Audio, 3)
            .modality(ModalityType::Sensor, 3)
            .fusion_strategy(FusionStrategy::CrossAttention)
            .build()
            .unwrap();

        pipeline.engine_mut().set_model(create_test_model(3, 3));

        let audio = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let sensor = Array1::from_vec(vec![0.4, 0.5, 0.6]);

        let result =
            pipeline.forward(&[(ModalityType::Audio, audio), (ModalityType::Sensor, sensor)]);

        assert!(result.is_ok());
    }

    #[test]
    fn test_hierarchical_fusion() {
        let engine_config = EngineConfig::new(4, 4);
        let mut pipeline = MultiModalPipeline::builder()
            .engine_config(engine_config)
            .modality(ModalityType::Audio, 2)
            .modality(ModalityType::Text, 2)
            .fusion_strategy(FusionStrategy::Hierarchical)
            .build()
            .unwrap();

        pipeline.engine_mut().set_model(create_test_model(4, 4));

        let audio = Array1::from_vec(vec![0.1, 0.2]);
        let text = Array1::from_vec(vec![0.3, 0.4]);

        let result = pipeline.forward(&[(ModalityType::Audio, audio), (ModalityType::Text, text)]);

        result.unwrap(); // Show error
    }

    /// Regression: `weighted_fusion` iterated the `HashMap` directly, so its
    /// segment order (and therefore which numbers ended up in which
    /// position of the output) depended on hash-bucket layout instead of
    /// modality name — unlike `early_fusion`, which already sorted.
    #[test]
    fn test_weighted_fusion_matches_name_sorted_order() {
        let engine_config = EngineConfig::new(5, 5);
        let mut pipeline = MultiModalPipeline::builder()
            .engine_config(engine_config)
            .modality(ModalityType::Audio, 2)
            .modality(ModalityType::Video, 3)
            .fusion_strategy(FusionStrategy::WeightedFusion)
            .build()
            .unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(ModalityType::Audio, Array1::from_vec(vec![10.0, 10.0]));
        inputs.insert(
            ModalityType::Video,
            Array1::from_vec(vec![20.0, 20.0, 20.0]),
        );

        // "audio" < "video" alphabetically, so audio's (equal-weight,
        // normalized) segment must come first, deterministically.
        let fused = pipeline
            .fuse(&inputs)
            .expect("weighted fusion must succeed");
        assert_eq!(fused.to_vec(), vec![5.0, 5.0, 10.0, 10.0, 10.0]);
    }

    /// Regression: `hierarchical_fusion` averages `early_fusion` (sorted by
    /// name) element-wise with `weighted_fusion` (previously unsorted). When
    /// the two fusions disagreed on ordering, this silently averaged one
    /// modality's samples with a *different* modality's — e.g. audio
    /// samples with video samples — rather than each modality with itself.
    #[test]
    fn test_hierarchical_fusion_segments_align_per_modality() {
        let engine_config = EngineConfig::new(5, 5);
        let mut pipeline = MultiModalPipeline::builder()
            .engine_config(engine_config)
            .modality(ModalityType::Audio, 2)
            .modality(ModalityType::Video, 3)
            .fusion_strategy(FusionStrategy::Hierarchical)
            .build()
            .unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(ModalityType::Audio, Array1::from_vec(vec![10.0, 10.0]));
        inputs.insert(
            ModalityType::Video,
            Array1::from_vec(vec![20.0, 20.0, 20.0]),
        );

        // early = [10,10,20,20,20]; weighted (equal weights) = [5,5,10,10,10];
        // hierarchical = (early + weighted) / 2 = [7.5,7.5,15,15,15].
        // A misaligned implementation that placed video first in `weighted`
        // would instead sum audio's early segment with video's weighted
        // segment (and vice versa), producing a different result.
        let fused = pipeline
            .fuse(&inputs)
            .expect("hierarchical fusion must succeed");
        let expected = [7.5_f32, 7.5, 15.0, 15.0, 15.0];
        for (got, want) in fused.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-5,
                "hierarchical fusion segments must align per modality: got {:?}, want {:?}",
                fused.to_vec(),
                expected
            );
        }
    }

    #[test]
    fn test_modality_preprocessor() {
        let engine_config = EngineConfig::new(3, 3);

        let preprocessor: ModalityPreprocessor = Arc::new(|input| Ok(input.mapv(|x| x * 2.0)));

        let config = ModalityConfig::new(ModalityType::Audio, 3).preprocessor(preprocessor);

        let mut pipeline = MultiModalPipeline::builder()
            .engine_config(engine_config)
            .add_modality(config)
            .build()
            .unwrap();

        pipeline.engine_mut().set_model(create_test_model(3, 3));

        let audio = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let result = pipeline.forward(&[(ModalityType::Audio, audio)]);

        assert!(result.is_ok());
    }
}
