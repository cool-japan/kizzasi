//! Stream processing pipeline for composable transformations
//!
//! This module provides a framework for building complex stream processing
//! pipelines with reusable, composable components.

use crate::error::{IoError, IoResult};
use async_trait::async_trait;
use scirs2_core::ndarray::Array1;

/// Trait for stream transformations that can be chained
#[async_trait]
pub trait StreamTransform: Send + Sync {
    /// Transform input data to output data
    async fn transform(&mut self, input: Array1<f32>) -> IoResult<Array1<f32>>;

    /// Optional processing after all data
    async fn finalize(&mut self) -> IoResult<()> {
        Ok(())
    }

    /// Get transform name for debugging
    fn name(&self) -> &str {
        "StreamTransform"
    }
}

/// A pipeline of stream transformations
pub struct Pipeline {
    transforms: Vec<Box<dyn StreamTransform>>,
}

impl Pipeline {
    /// Create a new empty pipeline
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Add a transform to the pipeline
    pub fn add_transform<T: StreamTransform + 'static>(mut self, transform: T) -> Self {
        self.transforms.push(Box::new(transform));
        self
    }

    /// Process data through all transforms in sequence
    pub async fn process(&mut self, mut data: Array1<f32>) -> IoResult<Array1<f32>> {
        for transform in &mut self.transforms {
            data = transform.transform(data).await?;
        }
        Ok(data)
    }

    /// Process batch of data
    pub async fn process_batch(&mut self, batch: Vec<Array1<f32>>) -> IoResult<Vec<Array1<f32>>> {
        let mut results = Vec::with_capacity(batch.len());
        for data in batch {
            results.push(self.process(data).await?);
        }
        Ok(results)
    }

    /// Finalize all transforms
    pub async fn finalize(&mut self) -> IoResult<()> {
        for transform in &mut self.transforms {
            transform.finalize().await?;
        }
        Ok(())
    }

    /// Get number of transforms
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Check if pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Scaling transform (multiply by constant)
pub struct ScaleTransform {
    scale: f32,
}

impl ScaleTransform {
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }
}

#[async_trait]
impl StreamTransform for ScaleTransform {
    async fn transform(&mut self, input: Array1<f32>) -> IoResult<Array1<f32>> {
        Ok(input * self.scale)
    }

    fn name(&self) -> &str {
        "ScaleTransform"
    }
}

/// Offset transform (add constant)
pub struct OffsetTransform {
    offset: f32,
}

impl OffsetTransform {
    pub fn new(offset: f32) -> Self {
        Self { offset }
    }
}

#[async_trait]
impl StreamTransform for OffsetTransform {
    async fn transform(&mut self, input: Array1<f32>) -> IoResult<Array1<f32>> {
        Ok(input + self.offset)
    }

    fn name(&self) -> &str {
        "OffsetTransform"
    }
}

/// Clipping transform (limit values to range)
pub struct ClipTransform {
    min: f32,
    max: f32,
}

impl ClipTransform {
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }
}

#[async_trait]
impl StreamTransform for ClipTransform {
    async fn transform(&mut self, input: Array1<f32>) -> IoResult<Array1<f32>> {
        Ok(input.mapv(|x| x.clamp(self.min, self.max)))
    }

    fn name(&self) -> &str {
        "ClipTransform"
    }
}

/// Normalization transform (scale to [0, 1] or [-1, 1])
pub struct NormalizeTransform {
    min_val: Option<f32>,
    max_val: Option<f32>,
    centered: bool,
}

impl NormalizeTransform {
    /// Create normalizer with auto-detection of min/max
    pub fn new(centered: bool) -> Self {
        Self {
            min_val: None,
            max_val: None,
            centered,
        }
    }

    /// Create normalizer with fixed min/max
    pub fn with_range(min: f32, max: f32, centered: bool) -> Self {
        Self {
            min_val: Some(min),
            max_val: Some(max),
            centered,
        }
    }
}

#[async_trait]
impl StreamTransform for NormalizeTransform {
    async fn transform(&mut self, input: Array1<f32>) -> IoResult<Array1<f32>> {
        let min = self
            .min_val
            .unwrap_or_else(|| input.iter().copied().fold(f32::INFINITY, f32::min));
        let max = self
            .max_val
            .unwrap_or_else(|| input.iter().copied().fold(f32::NEG_INFINITY, f32::max));

        let range = max - min;
        if range.abs() < 1e-10 {
            return Ok(input);
        }

        let normalized = if self.centered {
            // Scale to [-1, 1]
            (input - min) / range * 2.0 - 1.0
        } else {
            // Scale to [0, 1]
            (input - min) / range
        };

        Ok(normalized)
    }

    fn name(&self) -> &str {
        "NormalizeTransform"
    }
}

/// Decimation transform (downsample by factor)
pub struct DecimateTransform {
    factor: usize,
}

impl DecimateTransform {
    pub fn new(factor: usize) -> Self {
        assert!(factor > 0, "Decimation factor must be > 0");
        Self { factor }
    }
}

#[async_trait]
impl StreamTransform for DecimateTransform {
    async fn transform(&mut self, input: Array1<f32>) -> IoResult<Array1<f32>> {
        let output: Vec<f32> = input.iter().step_by(self.factor).copied().collect();
        Ok(Array1::from_vec(output))
    }

    fn name(&self) -> &str {
        "DecimateTransform"
    }
}

/// Moving average transform (smooth signal)
pub struct MovingAverageTransform {
    window_size: usize,
    buffer: Vec<f32>,
}

impl MovingAverageTransform {
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0, "Window size must be > 0");
        Self {
            window_size,
            buffer: Vec::with_capacity(window_size),
        }
    }
}

#[async_trait]
impl StreamTransform for MovingAverageTransform {
    async fn transform(&mut self, input: Array1<f32>) -> IoResult<Array1<f32>> {
        let mut output = Vec::with_capacity(input.len());

        for &val in input.iter() {
            self.buffer.push(val);
            if self.buffer.len() > self.window_size {
                self.buffer.remove(0);
            }

            let avg: f32 = self.buffer.iter().sum::<f32>() / self.buffer.len() as f32;
            output.push(avg);
        }

        Ok(Array1::from_vec(output))
    }

    fn name(&self) -> &str {
        "MovingAverageTransform"
    }
}

/// Derivative transform (compute differences)
pub struct DerivativeTransform {
    last_value: Option<f32>,
}

impl DerivativeTransform {
    pub fn new() -> Self {
        Self { last_value: None }
    }
}

impl Default for DerivativeTransform {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl StreamTransform for DerivativeTransform {
    async fn transform(&mut self, input: Array1<f32>) -> IoResult<Array1<f32>> {
        let mut output = Vec::with_capacity(input.len());

        for &val in input.iter() {
            if let Some(last) = self.last_value {
                output.push(val - last);
            } else {
                output.push(0.0);
            }
            self.last_value = Some(val);
        }

        Ok(Array1::from_vec(output))
    }

    fn name(&self) -> &str {
        "DerivativeTransform"
    }
}

/// Parallel pipeline - process multiple pipelines in parallel and combine results
pub struct ParallelPipeline {
    pipelines: Vec<Pipeline>,
    combiner: CombineStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum CombineStrategy {
    /// Average all outputs
    Average,
    /// Take maximum value across outputs
    Maximum,
    /// Take minimum value across outputs
    Minimum,
    /// Sum all outputs
    Sum,
}

impl ParallelPipeline {
    /// Create parallel pipeline with combiner strategy
    pub fn new(combiner: CombineStrategy) -> Self {
        Self {
            pipelines: Vec::new(),
            combiner,
        }
    }

    /// Add a pipeline to run in parallel
    pub fn add_pipeline(mut self, pipeline: Pipeline) -> Self {
        self.pipelines.push(pipeline);
        self
    }

    /// Process data through all pipelines in parallel
    pub async fn process(&mut self, data: Array1<f32>) -> IoResult<Array1<f32>> {
        if self.pipelines.is_empty() {
            return Ok(data);
        }

        let mut results = Vec::with_capacity(self.pipelines.len());

        // Process through each pipeline
        for pipeline in &mut self.pipelines {
            let result = pipeline.process(data.clone()).await?;
            results.push(result);
        }

        // Combine results based on strategy
        self.combine_results(results)
    }

    fn combine_results(&self, results: Vec<Array1<f32>>) -> IoResult<Array1<f32>> {
        if results.is_empty() {
            return Err(IoError::InvalidConfig("No results to combine".to_string()));
        }

        let len = results[0].len();
        let mut combined = Array1::zeros(len);

        match self.combiner {
            CombineStrategy::Average => {
                for result in &results {
                    combined += result;
                }
                combined /= results.len() as f32;
            }
            CombineStrategy::Maximum => {
                for i in 0..len {
                    let max = results
                        .iter()
                        .map(|r| r[i])
                        .fold(f32::NEG_INFINITY, f32::max);
                    combined[i] = max;
                }
            }
            CombineStrategy::Minimum => {
                for i in 0..len {
                    let min = results.iter().map(|r| r[i]).fold(f32::INFINITY, f32::min);
                    combined[i] = min;
                }
            }
            CombineStrategy::Sum => {
                for result in &results {
                    combined += result;
                }
            }
        }

        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scale_transform() {
        let mut transform = ScaleTransform::new(2.0);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let output = transform.transform(input).await.unwrap();
        assert_eq!(output.as_slice().unwrap(), &[2.0, 4.0, 6.0]);
    }

    #[tokio::test]
    async fn test_pipeline() {
        let mut pipeline = Pipeline::new()
            .add_transform(ScaleTransform::new(2.0))
            .add_transform(OffsetTransform::new(1.0));

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let output = pipeline.process(input).await.unwrap();
        assert_eq!(output.as_slice().unwrap(), &[3.0, 5.0, 7.0]);
    }

    #[tokio::test]
    async fn test_clip_transform() {
        let mut transform = ClipTransform::new(-1.0, 1.0);
        let input = Array1::from_vec(vec![-2.0, 0.5, 2.0]);
        let output = transform.transform(input).await.unwrap();
        assert_eq!(output.as_slice().unwrap(), &[-1.0, 0.5, 1.0]);
    }

    #[tokio::test]
    async fn test_normalize_transform() {
        let mut transform = NormalizeTransform::with_range(0.0, 10.0, false);
        let input = Array1::from_vec(vec![0.0, 5.0, 10.0]);
        let output = transform.transform(input).await.unwrap();
        assert_eq!(output.as_slice().unwrap(), &[0.0, 0.5, 1.0]);
    }
}
