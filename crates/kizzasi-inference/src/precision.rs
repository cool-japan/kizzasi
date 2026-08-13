//! Mixed precision support for inference
//!
//! This module provides FP16 (half precision) and BF16 (bfloat16) *numerics* for
//! inference: values are rounded to the selected reduced precision at the
//! boundaries of an operation, so a prediction matches what a reduced-precision
//! backend would produce and the accuracy impact of a precision choice can be
//! measured before committing to it.
//!
//! # What this module does — and does not — do
//!
//! [`PrecisionConverter::convert_and_compute_1d`] changes the *numerics*, not the
//! storage layout: arrays stay `Array1<f32>` between operations, so selecting FP16
//! or BF16 does not by itself halve resident memory or speed anything up. For
//! genuinely half-width storage use [`PrecisionConverter::to_fp16_1d`] /
//! [`PrecisionConverter::to_bf16_1d`], which return `Vec<f16>` / `Vec<bf16>`, or
//! [`PrecisionConverter::compute_fp16_1d`] / [`PrecisionConverter::compute_bf16_1d`],
//! which keep the data half-width across the operation.
//!
//! [`PrecisionMode::Mixed`] controls where the *result* lands: with
//! `accumulate_fp32 == true` the operation's output keeps full FP32 range and
//! resolution (an FP32 accumulator), with `accumulate_fp32 == false` the output is
//! rounded back to the compute precision as a reduced-precision accumulator would.

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

    /// Get the memory reduction factor of this mode, for data held in
    /// reduced-precision storage (`Vec<f16>` / `Vec<bf16>`)
    ///
    /// This is a property of the *mode*, not a measurement: it describes the
    /// footprint of tensors converted with [`PrecisionConverter::to_fp16_1d`] and
    /// friends. Values that stay in `Array1<f32>` are unaffected.
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
    /// Loss scaling factor applied before rounding to `FP16`/`BF16` and
    /// removed afterwards, to prevent small-magnitude values from
    /// underflowing the reduced format's subnormal range.
    ///
    /// This is the same pre-scale/post-scale mechanism mixed-precision
    /// *training* uses to protect small gradients from flushing to zero,
    /// applied here to whatever values this (inference-only) converter
    /// rounds. `<= 0.0` or non-finite is treated as `1.0` (no scaling).
    pub loss_scale: f32,
    /// Whether to automatically back off `loss_scale` (by halving) when it
    /// would overflow the target format for the data at hand, rather than
    /// producing `inf`.
    pub dynamic_loss_scale: bool,
    /// Maximum absolute value passed through to `FP16`/`BF16` rounding.
    ///
    /// Values outside `[-threshold, threshold]` are clamped *before*
    /// rounding, preventing a large outlier from overflowing to `inf` in the
    /// reduced format. This is magnitude clipping applied at the
    /// reduced-precision boundary, not gradient clipping — this converter is
    /// inference-only and has no gradients to clip.
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

    /// Run `op` with the operand — and, depending on the mode, the result — held at
    /// the configured reduced precision
    ///
    /// - `FP32`: `op` runs untouched.
    /// - `FP16` / `BF16`: operand *and* result are rounded to the reduced format,
    ///   which is what a backend that stores activations in that format produces.
    /// - `Mixed { accumulate_fp32: true }`: operand is rounded, the result keeps
    ///   full FP32 range and resolution (FP32 accumulator).
    /// - `Mixed { accumulate_fp32: false }`: operand and result are both rounded
    ///   (reduced-precision accumulator) — identical to the pure reduced modes.
    pub fn convert_and_compute_1d(
        &self,
        data: &Array1<f32>,
        op: impl Fn(&Array1<f32>) -> Array1<f32>,
    ) -> InferenceResult<Array1<f32>> {
        let (compute, round_result) = match self.config.mode {
            PrecisionMode::FP32 => return Ok(op(data)),
            PrecisionMode::FP16 => (ComputePrecision::FP16, true),
            PrecisionMode::BF16 => (ComputePrecision::BF16, true),
            PrecisionMode::Mixed {
                compute,
                accumulate_fp32,
            } => (compute, !accumulate_fp32),
        };

        let reduced = self.round_scaled_1d(data, compute);
        let result = op(&reduced);

        if round_result {
            Ok(self.round_scaled_1d(&result, compute))
        } else {
            Ok(result)
        }
    }

    /// Round every element to `compute` precision and back to `f32`
    ///
    /// This is the raw rounding step, with no loss-scale pre/post-scaling
    /// and no `grad_clip_threshold` clamping — see
    /// `PrecisionConverter::round_scaled_1d` for the version the engine and
    /// `convert_and_compute_1d` actually use.
    pub fn round_1d(&self, data: &Array1<f32>, compute: ComputePrecision) -> Array1<f32> {
        data.mapv(|x| Self::round_scalar(x, compute))
    }

    fn round_scalar(x: f32, compute: ComputePrecision) -> f32 {
        match compute {
            ComputePrecision::FP16 => f16::from_f32(x).to_f32(),
            ComputePrecision::BF16 => bf16::from_f32(x).to_f32(),
        }
    }

    /// Round `data` to `compute` precision, applying
    /// [`PrecisionConfig::grad_clip_threshold`] and
    /// [`PrecisionConfig::loss_scale`]/[`PrecisionConfig::dynamic_loss_scale`]
    /// around the rounding step.
    fn round_scaled_1d(&self, data: &Array1<f32>, compute: ComputePrecision) -> Array1<f32> {
        let clamped = self.clamp_for_precision(data);
        let scale = self.resolve_loss_scale(&clamped, compute);
        if scale == 1.0 {
            return self.round_1d(&clamped, compute);
        }
        clamped.mapv(|x| Self::round_scalar(x * scale, compute) / scale)
    }

    /// Clamp `data` to `[-threshold, threshold]` when
    /// [`PrecisionConfig::grad_clip_threshold`] is set, protecting a large
    /// outlier from overflowing to `inf` once rounded to `FP16`/`BF16`.
    fn clamp_for_precision(&self, data: &Array1<f32>) -> Array1<f32> {
        match self.config.grad_clip_threshold {
            Some(threshold) if threshold.is_finite() && threshold > 0.0 => {
                data.mapv(|x| x.clamp(-threshold, threshold))
            }
            _ => data.clone(),
        }
    }

    /// Resolve the loss-scale factor to actually use when rounding `data` to
    /// `compute` precision.
    ///
    /// When [`PrecisionConfig::dynamic_loss_scale`] is enabled, a configured
    /// [`PrecisionConfig::loss_scale`] that would overflow the target format
    /// for `data` (any element becomes non-finite after scaling) is halved
    /// repeatedly until it no longer overflows — the same automatic backoff
    /// a training loop's dynamic loss scaler performs, applied here per call
    /// since this converter is stateless across calls.
    fn resolve_loss_scale(&self, data: &Array1<f32>, compute: ComputePrecision) -> f32 {
        let mut scale = self.config.loss_scale;
        if !(scale.is_finite() && scale > 0.0) {
            return 1.0;
        }
        if !self.config.dynamic_loss_scale {
            return scale;
        }

        let overflows = |s: f32| -> bool {
            data.iter().any(|&x| {
                let scaled = x * s;
                !scaled.is_finite() || !Self::round_scalar(scaled, compute).is_finite()
            })
        };

        while scale > 1.0 && overflows(scale) {
            scale /= 2.0;
        }
        scale
    }

    /// Apply the configured precision to a value, consuming it
    ///
    /// This is the cheap path used by the inference engine on every step: `FP32`
    /// returns the array untouched (no allocation, no copy), every other mode
    /// rounds it to the configured format (applying `grad_clip_threshold` and
    /// `loss_scale`/`dynamic_loss_scale`, same as `convert_and_compute_1d`).
    pub fn apply_1d(&self, data: Array1<f32>) -> Array1<f32> {
        match self.config.mode {
            PrecisionMode::FP32 => data,
            PrecisionMode::FP16 => self.round_scaled_1d(&data, ComputePrecision::FP16),
            PrecisionMode::BF16 => self.round_scaled_1d(&data, ComputePrecision::BF16),
            PrecisionMode::Mixed { compute, .. } => self.round_scaled_1d(&data, compute),
        }
    }

    /// Run `op` over genuinely half-width FP16 storage
    ///
    /// Unlike [`PrecisionConverter::convert_and_compute_1d`], the operand handed to
    /// `op` is a `&[f16]` — half the bytes of the FP32 original — and the result is
    /// carried as `Vec<f16>` until it is widened for the caller.
    pub fn compute_fp16_1d(
        &self,
        data: &Array1<f32>,
        op: impl Fn(&[f16]) -> Vec<f16>,
    ) -> Array1<f32> {
        let reduced = self.to_fp16_1d(data);
        let result = op(&reduced);
        self.from_fp16_1d(&result)
    }

    /// Run `op` over genuinely half-width BF16 storage
    ///
    /// See [`PrecisionConverter::compute_fp16_1d`].
    pub fn compute_bf16_1d(
        &self,
        data: &Array1<f32>,
        op: impl Fn(&[bf16]) -> Vec<bf16>,
    ) -> Array1<f32> {
        let reduced = self.to_bf16_1d(data);
        let result = op(&reduced);
        self.from_bf16_1d(&result)
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

    /// Regression: the two `Mixed` branches used to be byte-identical, so
    /// `accumulate_fp32` provably could not change any result.
    #[test]
    fn test_mixed_accumulate_fp32_changes_result() {
        // 1.0001 is not representable in FP16 (eps ~= 9.8e-4 near 1.0), so it
        // survives only when the result is accumulated in FP32.
        let data = Array1::from_vec(vec![1.0_f32]);
        let op = |x: &Array1<f32>| x.mapv(|v| v + 1e-4);

        let accumulated = PrecisionConverter::new(PrecisionConfig::new().mixed_fp16(true))
            .convert_and_compute_1d(&data, op)
            .expect("fp32 accumulation must succeed");
        let truncated = PrecisionConverter::new(PrecisionConfig::new().mixed_fp16(false))
            .convert_and_compute_1d(&data, op)
            .expect("fp16 accumulation must succeed");

        assert!(
            (accumulated[0] - 1.0001).abs() < 1e-6,
            "FP32 accumulation must keep the increment, got {}",
            accumulated[0]
        );
        assert_eq!(
            truncated[0], 1.0,
            "FP16 accumulation must round the increment away"
        );
        assert_ne!(accumulated[0], truncated[0]);
    }

    /// Regression: reduced-precision modes must round the *result* too, otherwise
    /// selecting FP16 only adds input rounding error and models nothing.
    #[test]
    fn test_reduced_modes_round_the_result() {
        let data = Array1::from_vec(vec![1.0_f32]);
        let op = |x: &Array1<f32>| x.mapv(|v| v + 1e-4);

        for config in [PrecisionConfig::new().fp16(), PrecisionConfig::new().bf16()] {
            let mode = config.mode;
            let result = PrecisionConverter::new(config)
                .convert_and_compute_1d(&data, op)
                .expect("reduced precision must succeed");
            assert_eq!(result[0], 1.0, "{} must round the result", mode.name());
        }
    }

    #[test]
    fn test_apply_1d_is_identity_for_fp32() {
        let converter = PrecisionConverter::new(PrecisionConfig::new());
        let data = Array1::from_vec(vec![1.0001_f32, -2.5, 3.75]);
        let result = converter.apply_1d(data.clone());
        assert_eq!(result, data);
    }

    #[test]
    fn test_apply_1d_rounds_for_fp16() {
        let converter = PrecisionConverter::new(PrecisionConfig::new().fp16());
        let result = converter.apply_1d(Array1::from_vec(vec![1.0001_f32]));
        assert_eq!(result[0], 1.0);
    }

    #[test]
    fn test_compute_fp16_1d_keeps_half_width_storage() {
        let converter = PrecisionConverter::new(PrecisionConfig::new().fp16());
        let data = Array1::from_vec(vec![1.0_f32, 2.0, 3.0]);

        let expected_bytes = data.len() * std::mem::size_of::<f16>();
        let result = converter.compute_fp16_1d(&data, |reduced| {
            // The operand really is half-width storage, not a widened copy.
            assert_eq!(std::mem::size_of_val(reduced), expected_bytes);
            reduced
                .iter()
                .map(|&x| f16::from_f32(x.to_f32() * 2.0))
                .collect()
        });

        assert_eq!(result.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    /// Regression: `loss_scale` was accepted and stored but never applied —
    /// a tiny signal that underflows FP16 directly must survive when
    /// pre-scaled.
    #[test]
    fn test_loss_scale_prevents_underflow() {
        let unscaled = PrecisionConverter::new(PrecisionConfig::new().fp16())
            .apply_1d(Array1::from_vec(vec![1e-8_f32]));
        assert_eq!(
            unscaled[0], 0.0,
            "1e-8 must flush to zero under direct FP16 rounding (sanity check)"
        );

        let scaled = PrecisionConverter::new(PrecisionConfig::new().fp16().loss_scale(1.0e6))
            .apply_1d(Array1::from_vec(vec![1e-8_f32]));
        assert!(
            scaled[0] != 0.0 && (scaled[0] - 1e-8).abs() / 1e-8 < 0.1,
            "loss_scale=1e6 must keep 1e-8 non-zero with < 10% relative error, got {}",
            scaled[0]
        );
    }

    /// Regression: `dynamic_loss_scale` was accepted and stored but never
    /// used — a configured scale large enough to overflow FP16 must now be
    /// backed off automatically instead of producing `inf`.
    #[test]
    fn test_dynamic_loss_scale_backs_off_on_overflow() {
        // 1000.0 * 1e6 overflows FP16's ~65504 max; dynamic backoff must
        // halve the scale until it fits, rather than producing `inf`.
        let result = PrecisionConverter::new(
            PrecisionConfig::new()
                .fp16()
                .loss_scale(1.0e6)
                .dynamic_loss_scale(true),
        )
        .apply_1d(Array1::from_vec(vec![1000.0_f32]));

        assert!(
            result[0].is_finite(),
            "dynamic_loss_scale must back off an overflowing scale, got {}",
            result[0]
        );
        assert!(
            (result[0] - 1000.0).abs() / 1000.0 < 0.01,
            "the backed-off scale must still round-trip close to the true value, got {}",
            result[0]
        );

        // Without dynamic backoff, the same configuration overflows to inf.
        let no_backoff = PrecisionConverter::new(PrecisionConfig::new().fp16().loss_scale(1.0e6))
            .apply_1d(Array1::from_vec(vec![1000.0_f32]));
        assert!(
            !no_backoff[0].is_finite(),
            "sanity check: without dynamic_loss_scale this configuration should overflow"
        );
    }

    /// Regression: `grad_clip_threshold` was accepted and stored but never
    /// applied — a large outlier that would overflow FP16 to `inf` must now
    /// be clamped to the threshold first.
    #[test]
    fn test_grad_clip_threshold_prevents_overflow() {
        let clipped =
            PrecisionConverter::new(PrecisionConfig::new().fp16().grad_clip_threshold(100.0))
                .apply_1d(Array1::from_vec(vec![1.0e9_f32, -1.0e9_f32, 50.0_f32]));

        assert_eq!(
            clipped[0], 100.0,
            "large positive outlier must clamp to the threshold"
        );
        assert_eq!(
            clipped[1], -100.0,
            "large negative outlier must clamp to -threshold"
        );
        assert_eq!(
            clipped[2], 50.0,
            "a value already inside the threshold must be unaffected"
        );

        // Without the threshold, the same input overflows to inf.
        let unclipped = PrecisionConverter::new(PrecisionConfig::new().fp16())
            .apply_1d(Array1::from_vec(vec![1.0e9_f32]));
        assert!(
            !unclipped[0].is_finite(),
            "sanity check: without grad_clip_threshold this input should overflow"
        );
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
