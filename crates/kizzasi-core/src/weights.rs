//! Weight management for SSM models
//!
//! Provides functionality to load/save model weights in various formats:
//! - Safetensors (preferred format)
//! - PyTorch checkpoints (via conversion)
//! - Quantized weights (INT8)
//! - LoRA adapters

use crate::device::DeviceConfig;
use crate::error::{CoreError, CoreResult};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Weight format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// Safetensors format (recommended)
    SafeTensors,
    /// PyTorch checkpoint
    PyTorch,
    /// Quantized INT8
    QuantizedInt8,
}

/// Weight loading configuration
#[derive(Debug, Clone)]
pub struct WeightLoadConfig {
    /// Device configuration (CPU, or Metal with the `metal` feature)
    pub device_config: DeviceConfig,
    /// Whether to quantize weights on load
    pub quantize: bool,
    /// Strict mode (fail if any keys are missing)
    pub strict: bool,
}

impl Default for WeightLoadConfig {
    fn default() -> Self {
        Self {
            device_config: DeviceConfig::default(),
            quantize: false,
            strict: true,
        }
    }
}

impl WeightLoadConfig {
    /// Create device from configuration
    pub fn create_device(&self) -> CoreResult<Device> {
        self.device_config.create_device()
    }

    /// Get data type from configuration
    pub fn get_dtype(&self) -> DType {
        if self.device_config.use_fp16 {
            DType::F16
        } else {
            DType::F32
        }
    }
}

/// Weight loader for SSM models
pub struct WeightLoader {
    config: WeightLoadConfig,
}

impl WeightLoader {
    /// Create a new weight loader
    pub fn new(config: WeightLoadConfig) -> Self {
        Self { config }
    }

    /// Load weights from a safetensors file.
    ///
    /// Unlike a bare `varmap.load(path)`, this honours every field of
    /// [`WeightLoadConfig`]:
    ///
    /// - `device_config`: `varmap`'s existing [`candle_core::Var`]s are pinned
    ///   to whatever device they were created on -- `WeightLoadConfig`
    ///   cannot move them post-hoc. What this loader *can* honestly do is
    ///   verify the configured device actually matches, and refuse to
    ///   proceed with a clear error on a mismatch instead of silently
    ///   loading onto the pre-existing device as if `device_config` had no
    ///   meaning.
    /// - `strict` (default `true`): if any key the model's `varmap` needs is
    ///   missing from the file, fail with the full list of missing keys
    ///   instead of loading a partially-initialised model. When `false`,
    ///   missing keys are left at their previously-initialised values rather
    ///   than erroring (a genuinely different, more lenient behaviour than
    ///   the always-strict `VarMap::load`, which is used only when nothing
    ///   is missing).
    /// - `quantize`: when set, every loaded tensor is round-tripped through
    ///   [`Self::quantize_tensor`]/[`Self::dequantize_tensor`] (INT8), so
    ///   the in-memory weights carry the same precision loss a genuine INT8
    ///   deployment would see. Storage stays at the `VarMap`'s own dtype
    ///   (F32/F16) -- this does not shrink memory usage, only applies the
    ///   quantization error, since `Var` cannot change dtype in place.
    pub fn load_safetensors<P: AsRef<Path>>(&self, path: P, varmap: &mut VarMap) -> CoreResult<()> {
        let path = path.as_ref();

        // Open the file directly (rather than delegating straight to
        // `VarMap::load`) so we can inspect which keys it actually contains
        // and apply `strict`/`device_config` before touching any tensor.
        let safetensors_data = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
            .map_err(|e| {
                CoreError::WeightLoadError(format!(
                    "Failed to open safetensors file '{}': {}",
                    path.display(),
                    e
                ))
            })?;

        let file_keys: HashSet<String> = safetensors_data
            .tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect();

        let expected_device = self.config.create_device()?;

        let var_names: Vec<String> = {
            let data = varmap.data().lock().map_err(|_| {
                CoreError::WeightLoadError("WeightLoader: VarMap mutex poisoned".to_string())
            })?;

            // Fail fast, before loading any tensor data, if the configured
            // device doesn't match where the model's variables actually
            // live. `device_config` cannot move an already-constructed
            // `VarMap`'s variables (they were pinned to a device at
            // model-construction time), so the most honest thing this
            // loader can do with a mismatch is refuse rather than silently
            // load onto whatever device happened to be used before.
            for var in data.values() {
                if !var.device().same_device(&expected_device) {
                    return Err(CoreError::WeightLoadError(format!(
                        "WeightLoadConfig.device_config requests {:?}, but this VarMap's \
                         variables already live on {:?}. device_config cannot move an existing \
                         VarMap's variables -- construct the VarMap/VarBuilder on the configured \
                         device instead.",
                        expected_device,
                        var.device(),
                    )));
                }
            }

            data.keys().cloned().collect()
        };

        let missing: Vec<String> = var_names
            .iter()
            .filter(|name| !file_keys.contains(name.as_str()))
            .cloned()
            .collect();

        if self.config.strict && !missing.is_empty() {
            let mut sorted_missing = missing.clone();
            sorted_missing.sort();
            return Err(CoreError::WeightLoadError(format!(
                "strict weight load failed: {} key(s) required by the model are missing from '{}': {:?}",
                sorted_missing.len(),
                path.display(),
                sorted_missing
            )));
        }

        if missing.is_empty() {
            // Every variable the model needs is present in the file: the
            // upstream `VarMap::load` behaviour (mmap + load + assign each)
            // is exactly right and already well-tested.
            varmap.load(path).map_err(|e| {
                CoreError::WeightLoadError(format!("Failed to load safetensors: {}", e))
            })?;
        } else {
            // Non-strict partial load: only assign variables that ARE
            // present in the file, leaving the rest at their
            // previously-initialised values instead of hard-failing (which
            // is what `VarMap::load` would do the moment it hit the first
            // missing key).
            let data = varmap.data().lock().map_err(|_| {
                CoreError::WeightLoadError("WeightLoader: VarMap mutex poisoned".to_string())
            })?;
            for (name, var) in data.iter() {
                if file_keys.contains(name.as_str()) {
                    let tensor = safetensors_data.load(name, var.device()).map_err(|e| {
                        CoreError::WeightLoadError(format!(
                            "Failed to load tensor '{}': {}",
                            name, e
                        ))
                    })?;
                    var.set(&tensor).map_err(|e| {
                        CoreError::WeightLoadError(format!("Failed to set '{}': {}", name, e))
                    })?;
                }
            }
        }

        if self.config.quantize {
            let data = varmap.data().lock().map_err(|_| {
                CoreError::WeightLoadError("WeightLoader: VarMap mutex poisoned".to_string())
            })?;
            for var in data.values() {
                let q = self.quantize_tensor(var.as_tensor())?;
                let dequantized = self.dequantize_tensor(&q)?;
                var.set(&dequantized).map_err(|e| {
                    CoreError::WeightLoadError(format!(
                        "Failed to apply on-load quantization: {}",
                        e
                    ))
                })?;
            }
        }

        Ok(())
    }

    /// Save weights to a safetensors file
    ///
    /// Note: This function uses varmap.save() which handles saving to safetensors format
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P, varmap: &VarMap) -> CoreResult<()> {
        let path = path.as_ref();

        // Use VarMap's built-in safetensors saving
        varmap.save(path).map_err(|e| {
            CoreError::WeightLoadError(format!("Failed to save safetensors: {}", e))
        })?;

        Ok(())
    }

    /// Convert safetensors tensor view to candle Tensor
    #[allow(dead_code)]
    fn safetensors_to_candle(&self, view: safetensors::tensor::TensorView) -> CoreResult<Tensor> {
        let shape = view.shape().to_vec();
        let dtype = match view.dtype() {
            safetensors::Dtype::F32 => DType::F32,
            safetensors::Dtype::F16 => DType::F16,
            safetensors::Dtype::BF16 => DType::BF16,
            safetensors::Dtype::I64 => DType::I64,
            safetensors::Dtype::U8 => DType::U8,
            _ => {
                return Err(CoreError::WeightLoadError(format!(
                    "Unsupported dtype: {:?}",
                    view.dtype()
                )))
            }
        };

        // Get raw data
        let data = view.data();

        // Create tensor from raw data
        let tensor = match dtype {
            DType::F32 => {
                let values: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_vec(values, &shape[..], &Device::Cpu).map_err(|e| {
                    CoreError::WeightLoadError(format!("Failed to create tensor: {}", e))
                })?
            }
            DType::F16 | DType::BF16 => {
                // For F16/BF16, we need to convert to F32 first
                let values: Vec<u16> = data
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();

                let f32_values: Vec<f32> = values
                    .iter()
                    .map(|&v| half::f16::from_bits(v).to_f32())
                    .collect();

                Tensor::from_vec(f32_values, &shape[..], &Device::Cpu)
                    .map_err(|e| {
                        CoreError::WeightLoadError(format!("Failed to create tensor: {}", e))
                    })?
                    .to_dtype(dtype)
                    .map_err(|e| {
                        CoreError::WeightLoadError(format!("Failed to convert dtype: {}", e))
                    })?
            }
            _ => {
                return Err(CoreError::WeightLoadError(format!(
                    "Unsupported dtype for conversion: {:?}",
                    dtype
                )))
            }
        };

        Ok(tensor)
    }

    /// Convert candle tensors to safetensors format
    #[allow(dead_code)]
    fn candle_to_safetensors(&self, tensors: HashMap<String, Tensor>) -> CoreResult<Vec<u8>> {
        use safetensors::tensor::Dtype as SafeDtype;

        // Prepare tensor data
        let mut tensor_data: HashMap<String, (SafeDtype, Vec<usize>, Vec<u8>)> = HashMap::new();

        for (name, tensor) in tensors.iter() {
            let shape: Vec<usize> = tensor.dims().to_vec();

            let dtype = match tensor.dtype() {
                DType::F32 => SafeDtype::F32,
                DType::F16 => SafeDtype::F16,
                DType::BF16 => SafeDtype::BF16,
                DType::I64 => SafeDtype::I64,
                DType::U8 => SafeDtype::U8,
                _ => {
                    return Err(CoreError::WeightLoadError(format!(
                        "Unsupported dtype for safetensors: {:?}",
                        tensor.dtype()
                    )))
                }
            };

            // Get tensor data as bytes
            let data = self.tensor_to_bytes(tensor)?;

            tensor_data.insert(name.clone(), (dtype, shape, data));
        }

        // Build TensorView entries borrowing from the byte buffers in tensor_data.
        // The views are collected into a sorted BTreeMap so the header is deterministic.
        use std::collections::BTreeMap;
        let views: BTreeMap<String, safetensors::tensor::TensorView> = tensor_data
            .iter()
            .map(|(name, (dtype, shape, data))| {
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), data.as_slice())
                    .map(|view| (name.clone(), view))
                    .map_err(|e| {
                        CoreError::WeightLoadError(format!(
                            "Failed to create TensorView for '{}': {}",
                            name, e
                        ))
                    })
            })
            .collect::<CoreResult<_>>()?;

        safetensors::serialize(&views, None).map_err(|e| {
            CoreError::WeightLoadError(format!("Failed to serialize safetensors: {}", e))
        })
    }

    /// Convert tensor to bytes
    #[allow(dead_code)]
    fn tensor_to_bytes(&self, tensor: &Tensor) -> CoreResult<Vec<u8>> {
        match tensor.dtype() {
            DType::F32 => {
                let values = tensor
                    .flatten_all()
                    .map_err(|e| {
                        CoreError::WeightLoadError(format!("Failed to flatten tensor: {}", e))
                    })?
                    .to_vec1::<f32>()
                    .map_err(|e| {
                        CoreError::WeightLoadError(format!("Failed to convert to vec: {}", e))
                    })?;

                let mut bytes = Vec::with_capacity(values.len() * 4);
                for v in values {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                Ok(bytes)
            }
            DType::F16 => {
                let values = tensor
                    .flatten_all()
                    .map_err(|e| {
                        CoreError::WeightLoadError(format!("Failed to flatten tensor: {}", e))
                    })?
                    .to_vec1::<half::f16>()
                    .map_err(|e| {
                        CoreError::WeightLoadError(format!("Failed to convert to vec: {}", e))
                    })?;

                let mut bytes = Vec::with_capacity(values.len() * 2);
                for v in values {
                    bytes.extend_from_slice(&v.to_bits().to_le_bytes());
                }
                Ok(bytes)
            }
            _ => Err(CoreError::WeightLoadError(format!(
                "Unsupported dtype for bytes conversion: {:?}",
                tensor.dtype()
            ))),
        }
    }

    /// Quantize a candle [`Tensor`] to INT8 with per-tensor affine parameters.
    ///
    /// Computes per-tensor `min` and `max`, derives an affine
    /// `(scale, zero_point)` pair that maps the original range into `[0, 255]`,
    /// and stores the quantized payload as a `U8` tensor alongside the
    /// reconstruction parameters in a [`QuantizedTensor`].
    ///
    /// Quantization formula:
    /// ```text
    /// q = round(x / scale + zero_point).clamp(0, 255) as u8
    /// scale = max(max - min, EPS) / 255
    /// zero_point = round(-min / scale).clamp(0, 255)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::WeightLoadError`] if any candle tensor op fails
    /// (e.g. scalar extraction, dtype conversion).
    pub fn quantize_tensor(&self, tensor: &Tensor) -> CoreResult<QuantizedTensor> {
        let original_dtype = tensor.dtype();
        let original_shape = tensor.dims().to_vec();

        // Cast to f32 for scalar reductions; min/max on integer or fp16 dtypes
        // would force the caller to deal with dtype-specific scalar extraction.
        let tensor_f32 = tensor
            .to_dtype(DType::F32)
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to cast to f32: {}", e)))?;

        let (min_val, max_val) = tensor_scalar_min_max(&tensor_f32)?;

        let (scale, zero_point) = compute_affine_qparams(min_val, max_val);

        // Quantize: q = round(x * (1/scale) + zero_point).
        // candle's `affine(mul, add)` performs `x * mul + add` element-wise.
        let inv_scale = 1.0_f32 / scale;
        let scaled = tensor_f32
            .affine(inv_scale as f64, zero_point as f64)
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to scale: {}", e)))?
            .round()
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to round: {}", e)))?
            .clamp(0f64, 255f64)
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to clamp: {}", e)))?;

        let data = scaled
            .to_dtype(DType::U8)
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to convert to U8: {}", e)))?;

        Ok(QuantizedTensor {
            data,
            scale,
            zero_point,
            original_dtype,
            original_shape,
        })
    }

    /// Dequantize a [`QuantizedTensor`] back to a floating-point tensor.
    ///
    /// Reconstruction formula:
    /// ```text
    /// x = (q as f32 - zero_point) * scale
    /// ```
    ///
    /// The result is cast back to [`QuantizedTensor::original_dtype`] so the
    /// returned tensor is byte-compatible with the original signature.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::WeightLoadError`] if any candle tensor op fails.
    pub fn dequantize_tensor(&self, q: &QuantizedTensor) -> CoreResult<Tensor> {
        let scale = q.scale as f64;
        let zero_point = q.zero_point as f64;

        // Cast U8 -> F32, then apply x = (q - zp) * scale  =>  q * scale + (-zp * scale).
        let f = q
            .data
            .to_dtype(DType::F32)
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to cast q to f32: {}", e)))?
            .affine(scale, -zero_point * scale)
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to dequantize: {}", e)))?;

        f.to_dtype(q.original_dtype).map_err(|e| {
            CoreError::WeightLoadError(format!("Failed to restore original dtype: {}", e))
        })
    }

    /// Quantize a candle [`Tensor`] to INT8 with per-channel affine parameters
    /// along the given `axis`.
    ///
    /// This produces a much tighter quantization error than per-tensor
    /// quantization when the value distribution differs strongly between
    /// channels (e.g. INT8 matmul weights for transformer / SSM projections).
    ///
    /// Implementation strategy: slice the input along `axis` using
    /// [`Tensor::narrow`], quantize each slice independently with its own
    /// `(scale, zero_point)`, then re-assemble the U8 slices with
    /// [`Tensor::cat`] on the same axis.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidConfig`] if `axis` is out of bounds, or
    /// [`CoreError::WeightLoadError`] if any candle tensor op fails.
    pub fn quantize_per_channel(
        &self,
        tensor: &Tensor,
        axis: usize,
    ) -> CoreResult<PerChannelQuantizedTensor> {
        let original_dtype = tensor.dtype();
        let original_shape = tensor.dims().to_vec();

        if axis >= original_shape.len() {
            return Err(CoreError::InvalidConfig(format!(
                "quantize_per_channel: axis {} out of bounds for shape {:?}",
                axis, original_shape
            )));
        }

        let tensor_f32 = tensor
            .to_dtype(DType::F32)
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to cast to f32: {}", e)))?;

        let num_channels = original_shape[axis];
        let mut scales: Vec<f32> = Vec::with_capacity(num_channels);
        let mut zero_points: Vec<i32> = Vec::with_capacity(num_channels);
        let mut quantized_slices: Vec<Tensor> = Vec::with_capacity(num_channels);

        for i in 0..num_channels {
            let slice = tensor_f32.narrow(axis, i, 1).map_err(|e| {
                CoreError::WeightLoadError(format!(
                    "Failed to narrow channel {} on axis {}: {}",
                    i, axis, e
                ))
            })?;

            let (slice_min, slice_max) = tensor_scalar_min_max(&slice)?;
            let (scale, zero_point) = compute_affine_qparams(slice_min, slice_max);
            let inv_scale = 1.0_f32 / scale;

            let q_slice = slice
                .affine(inv_scale as f64, zero_point as f64)
                .map_err(|e| {
                    CoreError::WeightLoadError(format!("Failed to scale channel {}: {}", i, e))
                })?
                .round()
                .map_err(|e| {
                    CoreError::WeightLoadError(format!("Failed to round channel {}: {}", i, e))
                })?
                .clamp(0f64, 255f64)
                .map_err(|e| {
                    CoreError::WeightLoadError(format!("Failed to clamp channel {}: {}", i, e))
                })?
                .to_dtype(DType::U8)
                .map_err(|e| {
                    CoreError::WeightLoadError(format!("Failed to cast channel {} to U8: {}", i, e))
                })?;

            scales.push(scale);
            zero_points.push(zero_point);
            quantized_slices.push(q_slice);
        }

        // Re-assemble the per-channel slices back into a single U8 tensor.
        let data = Tensor::cat(&quantized_slices, axis)
            .map_err(|e| CoreError::WeightLoadError(format!("Failed to cat slices: {}", e)))?;

        Ok(PerChannelQuantizedTensor {
            data,
            scales,
            zero_points,
            axis,
            original_dtype,
            original_shape,
        })
    }

    /// Dequantize a [`PerChannelQuantizedTensor`] back to a floating-point
    /// tensor by applying each channel's affine parameters independently.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidConfig`] if the channel metadata is
    /// inconsistent, or [`CoreError::WeightLoadError`] if any candle tensor op
    /// fails.
    pub fn dequantize_per_channel(&self, q: &PerChannelQuantizedTensor) -> CoreResult<Tensor> {
        if q.axis >= q.original_shape.len() {
            return Err(CoreError::InvalidConfig(format!(
                "dequantize_per_channel: axis {} out of bounds for shape {:?}",
                q.axis, q.original_shape
            )));
        }

        let num_channels = q.original_shape[q.axis];
        if q.scales.len() != num_channels || q.zero_points.len() != num_channels {
            return Err(CoreError::InvalidConfig(format!(
                "dequantize_per_channel: expected {} scales/zero_points, got {}/{}",
                num_channels,
                q.scales.len(),
                q.zero_points.len()
            )));
        }

        let mut dequantized_slices: Vec<Tensor> = Vec::with_capacity(num_channels);

        for i in 0..num_channels {
            let slice = q.data.narrow(q.axis, i, 1).map_err(|e| {
                CoreError::WeightLoadError(format!(
                    "Failed to narrow channel {} on axis {}: {}",
                    i, q.axis, e
                ))
            })?;

            let scale = q.scales[i] as f64;
            let zero_point = q.zero_points[i] as f64;

            let f = slice
                .to_dtype(DType::F32)
                .map_err(|e| {
                    CoreError::WeightLoadError(format!(
                        "Failed to cast channel {} to f32: {}",
                        i, e
                    ))
                })?
                .affine(scale, -zero_point * scale)
                .map_err(|e| {
                    CoreError::WeightLoadError(format!("Failed to dequantize channel {}: {}", i, e))
                })?;

            dequantized_slices.push(f);
        }

        let result = Tensor::cat(&dequantized_slices, q.axis).map_err(|e| {
            CoreError::WeightLoadError(format!("Failed to cat dequantized slices: {}", e))
        })?;

        result.to_dtype(q.original_dtype).map_err(|e| {
            CoreError::WeightLoadError(format!("Failed to restore original dtype: {}", e))
        })
    }
}

/// INT8-quantized tensor with per-tensor affine reconstruction parameters.
///
/// Quantization:   `q = round(x / scale + zero_point).clamp(0, 255) as u8`
/// Dequantization: `x = (q as f32 - zero_point as f32) * scale`
///
/// `scale` is always finite and strictly positive; `zero_point` is constrained
/// to the range `[0, 255]` so that a value of `min` round-trips through
/// `q = 0` exactly (modulo `scale * round` error).
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized payload (`DType::U8`).
    pub data: Tensor,
    /// Per-tensor scale factor (always `> 0` and finite).
    pub scale: f32,
    /// Per-tensor zero point in `[0, 255]`.
    pub zero_point: i32,
    /// Original tensor dtype (for round-tripping).
    pub original_dtype: DType,
    /// Original tensor shape (for round-tripping).
    pub original_shape: Vec<usize>,
}

/// INT8-quantized tensor with **per-channel** affine reconstruction parameters
/// along a single `axis`.
///
/// Each entry of `scales` / `zero_points` corresponds to one slice of the
/// quantized data along `axis`. For weight matrices `[out, in]` quantized
/// along `axis = 0`, every output channel gets its own `(scale, zero_point)`,
/// which is the standard accuracy-preserving form for INT8 matmul.
#[derive(Debug, Clone)]
pub struct PerChannelQuantizedTensor {
    /// Quantized payload (`DType::U8`).
    pub data: Tensor,
    /// One scale factor per channel along `axis`.
    pub scales: Vec<f32>,
    /// One zero point per channel along `axis`.
    pub zero_points: Vec<i32>,
    /// Axis along which the per-channel parameters are indexed.
    pub axis: usize,
    /// Original tensor dtype (for round-tripping).
    pub original_dtype: DType,
    /// Original tensor shape (for round-tripping).
    pub original_shape: Vec<usize>,
}

/// Minimum allowed range to avoid divide-by-zero on constant tensors.
const QUANT_RANGE_EPS: f32 = 1e-12;

/// Compute the per-tensor scalar min and max of an `f32` tensor.
fn tensor_scalar_min_max(tensor: &Tensor) -> CoreResult<(f32, f32)> {
    let flat = tensor
        .flatten_all()
        .map_err(|e| CoreError::WeightLoadError(format!("Failed to flatten tensor: {}", e)))?;

    let min_scalar = flat
        .min(0)
        .map_err(|e| CoreError::WeightLoadError(format!("Failed to compute min: {}", e)))?
        .to_scalar::<f32>()
        .map_err(|e| CoreError::WeightLoadError(format!("Failed to extract min scalar: {}", e)))?;

    let max_scalar = flat
        .max(0)
        .map_err(|e| CoreError::WeightLoadError(format!("Failed to compute max: {}", e)))?
        .to_scalar::<f32>()
        .map_err(|e| CoreError::WeightLoadError(format!("Failed to extract max scalar: {}", e)))?;

    Ok((min_scalar, max_scalar))
}

/// Compute affine `(scale, zero_point)` from a value range.
///
/// Always returns a finite, strictly positive `scale` and a `zero_point` in
/// `[0, 255]`.
fn compute_affine_qparams(min_val: f32, max_val: f32) -> (f32, i32) {
    let range = (max_val - min_val).max(QUANT_RANGE_EPS);
    let scale = range / 255.0_f32;
    let zero_point_f = (-min_val / scale).round().clamp(0.0_f32, 255.0_f32);
    (scale, zero_point_f as i32)
}

/// Weight pruning utilities
pub struct WeightPruner;

impl WeightPruner {
    /// Prune weights by magnitude
    ///
    /// Sets weights with absolute value below threshold to zero.
    pub fn prune_by_magnitude(tensor: &Tensor, threshold: f32) -> CoreResult<Tensor> {
        let abs_tensor = tensor
            .abs()
            .map_err(|e| CoreError::Generic(format!("Failed to compute abs: {}", e)))?;

        let mask = abs_tensor
            .ge(threshold as f64)
            .map_err(|e| CoreError::Generic(format!("Failed to create mask: {}", e)))?
            .to_dtype(tensor.dtype())
            .map_err(|e| CoreError::Generic(format!("Failed to convert mask dtype: {}", e)))?;

        tensor
            .mul(&mask)
            .map_err(|e| CoreError::Generic(format!("Failed to apply mask: {}", e)))
    }

    /// Prune weights by percentage
    ///
    /// Keeps only the top (1 - percentage) weights by magnitude.
    pub fn prune_by_percentage(tensor: &Tensor, percentage: f32) -> CoreResult<Tensor> {
        if percentage <= 0.0 || percentage >= 1.0 {
            return Err(CoreError::InvalidConfig(
                "Percentage must be between 0 and 1".to_string(),
            ));
        }

        // Flatten tensor to 1D for sorting
        let flat = tensor
            .flatten_all()
            .map_err(|e| CoreError::Generic(format!("Failed to flatten: {}", e)))?;

        let abs_flat = flat
            .abs()
            .map_err(|e| CoreError::Generic(format!("Failed to compute abs: {}", e)))?;

        // Get values as vec
        let values = abs_flat
            .to_vec1::<f32>()
            .map_err(|e| CoreError::Generic(format!("Failed to convert to vec: {}", e)))?;

        // Find threshold at the given percentage
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let threshold_idx = (sorted_values.len() as f32 * percentage) as usize;
        let threshold = sorted_values[threshold_idx];

        Self::prune_by_magnitude(tensor, threshold)
    }

    /// Compute sparsity of a tensor
    pub fn compute_sparsity(tensor: &Tensor) -> CoreResult<f32> {
        let total_elements = tensor.elem_count();

        let zeros = tensor
            .eq(0.0)
            .map_err(|e| CoreError::Generic(format!("Failed to compare with zero: {}", e)))?
            .to_dtype(DType::F32)
            .map_err(|e| CoreError::Generic(format!("Failed to convert dtype: {}", e)))?
            .sum_all()
            .map_err(|e| CoreError::Generic(format!("Failed to sum: {}", e)))?
            .to_vec0::<f32>()
            .map_err(|e| CoreError::Generic(format!("Failed to extract value: {}", e)))?;

        Ok(zeros / total_elements as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarBuilder;

    #[test]
    fn test_weight_loader_creation() {
        let config = WeightLoadConfig::default();
        let _loader = WeightLoader::new(config);
    }

    #[test]
    fn test_prune_by_magnitude() {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[1.0f32, 0.1, 2.0, 0.05, 3.0], &device).unwrap();

        let pruned = WeightPruner::prune_by_magnitude(&tensor, 0.5).unwrap();
        let values = pruned.to_vec1::<f32>().unwrap();

        assert_eq!(values, vec![1.0, 0.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn test_compute_sparsity() {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[1.0f32, 0.0, 2.0, 0.0, 3.0], &device).unwrap();

        let sparsity = WeightPruner::compute_sparsity(&tensor).unwrap();
        assert!((sparsity - 0.4).abs() < 1e-5);
    }

    #[test]
    fn test_safetensors_roundtrip() {
        use std::env;

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Create some test variables
        let _w1 = vb
            .get_with_hints((3, 4), "weight1", candle_nn::init::Init::Const(1.0))
            .unwrap();
        let _w2 = vb
            .get_with_hints((5, 6), "weight2", candle_nn::init::Init::Const(2.0))
            .unwrap();

        let config = WeightLoadConfig::default();
        let loader = WeightLoader::new(config);

        // Save
        let temp_dir = env::temp_dir();
        let save_path = temp_dir.join("test_weights.safetensors");

        let result = loader.save_safetensors(&save_path, &varmap);
        assert!(result.is_ok());

        // Clean up
        if save_path.exists() {
            std::fs::remove_file(save_path).ok();
        }
    }

    // ------------------------------------------------------------------
    // Regression tests: `WeightLoadConfig.strict`/`quantize` used to be
    // stored on `WeightLoader` and never read (`#[allow(dead_code)]`);
    // `load_safetensors` delegated straight to `varmap.load()` regardless
    // of configuration.
    // ------------------------------------------------------------------

    /// Save a two-variable VarMap (`w1`: (3,4) filled with 1.0, `w2`: (2,2)
    /// filled with 2.0) to a fresh temp file and return the path.
    fn save_two_var_checkpoint(filename: &str) -> std::path::PathBuf {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        vb.get_with_hints((3, 4), "w1", candle_nn::init::Init::Const(1.0))
            .unwrap();
        vb.get_with_hints((2, 2), "w2", candle_nn::init::Init::Const(2.0))
            .unwrap();

        let path = std::env::temp_dir().join(filename);
        WeightLoader::new(WeightLoadConfig::default())
            .save_safetensors(&path, &varmap)
            .unwrap();
        path
    }

    #[test]
    fn test_load_safetensors_strict_missing_key_errors() {
        let path = save_two_var_checkpoint("kizzasi_core_test_strict_missing.safetensors");

        // Fresh VarMap the "model" needs 3 keys, but the file only has 2:
        // `w3` is genuinely missing.
        let device = Device::Cpu;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        vb.get_with_hints((3, 4), "w1", candle_nn::init::Init::Const(0.0))
            .unwrap();
        vb.get_with_hints((2, 2), "w2", candle_nn::init::Init::Const(0.0))
            .unwrap();
        vb.get_with_hints((1, 1), "w3", candle_nn::init::Init::Const(0.0))
            .unwrap();

        let loader = WeightLoader::new(WeightLoadConfig::default()); // strict: true
        let result = loader.load_safetensors(&path, &mut varmap);
        assert!(result.is_err(), "strict load must fail on a missing key");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("w3"),
            "error message should name the missing key 'w3': {msg}"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_safetensors_non_strict_partial_load() {
        let path = save_two_var_checkpoint("kizzasi_core_test_nonstrict_partial.safetensors");

        let device = Device::Cpu;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        vb.get_with_hints((3, 4), "w1", candle_nn::init::Init::Const(0.0))
            .unwrap();
        vb.get_with_hints((2, 2), "w2", candle_nn::init::Init::Const(0.0))
            .unwrap();
        vb.get_with_hints((1, 1), "w3", candle_nn::init::Init::Const(9.5))
            .unwrap();

        let config = WeightLoadConfig {
            strict: false,
            ..WeightLoadConfig::default()
        };
        let loader = WeightLoader::new(config);
        loader
            .load_safetensors(&path, &mut varmap)
            .expect("non-strict load must succeed despite the missing 'w3' key");

        // w1/w2 were present in the file and must now hold the saved values.
        let data = varmap.data().lock().unwrap();
        let w1_val = data
            .get("w1")
            .unwrap()
            .as_tensor()
            .to_vec2::<f32>()
            .unwrap();
        assert!(w1_val
            .iter()
            .all(|row| row.iter().all(|&v| (v - 1.0).abs() < 1e-6)));
        let w2_val = data
            .get("w2")
            .unwrap()
            .as_tensor()
            .to_vec2::<f32>()
            .unwrap();
        assert!(w2_val
            .iter()
            .all(|row| row.iter().all(|&v| (v - 2.0).abs() < 1e-6)));

        // w3 was absent from the file and must be left at its initial value.
        let w3_val = data
            .get("w3")
            .unwrap()
            .as_tensor()
            .to_vec2::<f32>()
            .unwrap();
        assert!(
            (w3_val[0][0] - 9.5).abs() < 1e-6,
            "w3 must be untouched by a non-strict load, got {}",
            w3_val[0][0]
        );

        drop(data);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_safetensors_roundtrip_exact_values() {
        let path = save_two_var_checkpoint("kizzasi_core_test_load_roundtrip.safetensors");

        let device = Device::Cpu;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        vb.get_with_hints((3, 4), "w1", candle_nn::init::Init::Const(0.0))
            .unwrap();
        vb.get_with_hints((2, 2), "w2", candle_nn::init::Init::Const(0.0))
            .unwrap();

        let loader = WeightLoader::new(WeightLoadConfig::default());
        loader.load_safetensors(&path, &mut varmap).unwrap();

        let data = varmap.data().lock().unwrap();
        let w1_val = data
            .get("w1")
            .unwrap()
            .as_tensor()
            .to_vec2::<f32>()
            .unwrap();
        for row in &w1_val {
            for &v in row {
                assert_eq!(v, 1.0);
            }
        }
        let w2_val = data
            .get("w2")
            .unwrap()
            .as_tensor()
            .to_vec2::<f32>()
            .unwrap();
        for row in &w2_val {
            for &v in row {
                assert_eq!(v, 2.0);
            }
        }

        drop(data);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_safetensors_quantize_applies_int8_precision_loss() {
        // A non-constant tensor, so INT8 quantization has a real (nonzero)
        // rounding error to introduce -- a constant tensor round-trips
        // exactly regardless of quantization and wouldn't distinguish
        // "quantize was applied" from "quantize was silently ignored".
        let device = Device::Cpu;
        let varmap_save = VarMap::new();
        let vb_save = VarBuilder::from_varmap(&varmap_save, DType::F32, &device);
        let original = vb_save
            .get_with_hints(
                (16, 16),
                "w",
                candle_nn::init::Init::Randn {
                    mean: 0.0,
                    stdev: 1.0,
                },
            )
            .unwrap();
        let original_values = original.to_vec2::<f32>().unwrap();

        let path = std::env::temp_dir().join("kizzasi_core_test_quantize_load.safetensors");
        WeightLoader::new(WeightLoadConfig::default())
            .save_safetensors(&path, &varmap_save)
            .unwrap();

        let mut varmap_load = VarMap::new();
        let vb_load = VarBuilder::from_varmap(&varmap_load, DType::F32, &device);
        vb_load
            .get_with_hints((16, 16), "w", candle_nn::init::Init::Const(0.0))
            .unwrap();

        let config = WeightLoadConfig {
            quantize: true,
            ..WeightLoadConfig::default()
        };
        let loader = WeightLoader::new(config);
        loader.load_safetensors(&path, &mut varmap_load).unwrap();

        let data = varmap_load.data().lock().unwrap();
        let loaded_values = data.get("w").unwrap().as_tensor().to_vec2::<f32>().unwrap();

        // Must be CLOSE (INT8 over a randn tensor is a fine-grained
        // approximation) but not exact -- proving `quantize` actually ran
        // rather than being silently ignored (in which case the values
        // would match `original_values` bit-for-bit).
        let mut max_err = 0.0f32;
        let mut any_exact_mismatch = false;
        for i in 0..16 {
            for j in 0..16 {
                let err = (loaded_values[i][j] - original_values[i][j]).abs();
                max_err = max_err.max(err);
                if loaded_values[i][j] != original_values[i][j] {
                    any_exact_mismatch = true;
                }
            }
        }
        assert!(
            any_exact_mismatch,
            "quantize=true must perturb at least one value versus the unquantized original"
        );
        assert!(
            max_err < 0.2,
            "INT8 quantization error should be small for a randn tensor, got max_err={max_err}"
        );

        drop(data);
        std::fs::remove_file(&path).ok();
    }

    // ------------------------------------------------------------------
    // INT8 quantization tests (Track B)
    // ------------------------------------------------------------------

    /// Build a fresh `WeightLoader` for tests.
    fn test_loader() -> WeightLoader {
        WeightLoader::new(WeightLoadConfig::default())
    }

    /// Compute mean-squared-error between two `f32` tensors of the same shape.
    fn mse(a: &Tensor, b: &Tensor) -> f32 {
        let af = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let bf = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let av = af.to_vec1::<f32>().unwrap();
        let bv = bf.to_vec1::<f32>().unwrap();
        assert_eq!(av.len(), bv.len());
        let n = av.len() as f32;
        av.iter()
            .zip(bv.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>()
            / n
    }

    #[test]
    fn test_quantize_tensor_roundtrip() {
        let device = Device::Cpu;
        let loader = test_loader();

        // Normal-distributed input, shape [128, 64].
        let tensor = Tensor::randn(0.0_f32, 1.0_f32, (128, 64), &device).unwrap();

        let q = loader.quantize_tensor(&tensor).unwrap();
        assert_eq!(q.data.dtype(), DType::U8);
        assert!(q.scale.is_finite());
        assert!(q.scale > 0.0);
        assert!((0..=255).contains(&q.zero_point));

        let recovered = loader.dequantize_tensor(&q).unwrap();
        let err = mse(&tensor, &recovered);
        assert!(err < 1e-3, "MSE too high after roundtrip: {}", err);
    }

    #[test]
    fn test_quantize_zero_tensor() {
        let device = Device::Cpu;
        let loader = test_loader();

        let tensor = Tensor::zeros((16, 8), DType::F32, &device).unwrap();
        let q = loader.quantize_tensor(&tensor).unwrap();

        // Scale must remain finite and strictly positive even when range == 0.
        assert!(q.scale.is_finite(), "scale is not finite: {}", q.scale);
        assert!(q.scale > 0.0, "scale must be positive: {}", q.scale);
        // For an all-zero tensor, -min == 0, so zero_point should be 0.
        assert_eq!(q.zero_point, 0);

        let recovered = loader.dequantize_tensor(&q).unwrap();
        let values = recovered.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in values {
            assert_eq!(v, 0.0, "expected zero, got {}", v);
        }
    }

    #[test]
    fn test_quantize_positive_only() {
        let device = Device::Cpu;
        let loader = test_loader();

        // All values in [0, 1]
        let values: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        let tensor = Tensor::from_vec(values, &[256], &device).unwrap();
        let q = loader.quantize_tensor(&tensor).unwrap();

        // min == 0 => -min/scale == 0 => zero_point ~= 0.
        assert_eq!(
            q.zero_point, 0,
            "expected zero_point near 0 for non-negative input, got {}",
            q.zero_point
        );

        // Round-trip should be very accurate (linear ramp, 256 levels).
        let recovered = loader.dequantize_tensor(&q).unwrap();
        let err = mse(&tensor, &recovered);
        assert!(err < 1e-5, "MSE too high: {}", err);
    }

    #[test]
    fn test_quantize_negative_only() {
        let device = Device::Cpu;
        let loader = test_loader();

        // All values in [-1, 0]
        let values: Vec<f32> = (0..256).map(|i| -(i as f32) / 255.0).collect();
        let tensor = Tensor::from_vec(values, &[256], &device).unwrap();
        let q = loader.quantize_tensor(&tensor).unwrap();

        // min == -1, scale ~= 1/255, zero_point = round(1 / (1/255)) = 255.
        assert_eq!(
            q.zero_point, 255,
            "expected zero_point near 255 for non-positive input, got {}",
            q.zero_point
        );

        let recovered = loader.dequantize_tensor(&q).unwrap();
        let err = mse(&tensor, &recovered);
        assert!(err < 1e-5, "MSE too high: {}", err);
    }

    #[test]
    fn test_per_channel_vs_per_tensor_accuracy() {
        let device = Device::Cpu;
        let loader = test_loader();

        // Heteroscedastic input:
        //   row 0 has values in roughly [0, 1]   (small range)
        //   row 1 has values in roughly [0, 100] (large range)
        // Per-tensor quantization must use one global scale tuned for the
        // larger range, which destroys the precision of row 0.
        let mut values: Vec<f32> = Vec::with_capacity(2 * 128);
        for i in 0..128 {
            values.push(i as f32 / 127.0); // row 0 in [0, 1]
        }
        for i in 0..128 {
            values.push((i as f32 / 127.0) * 100.0); // row 1 in [0, 100]
        }
        let tensor = Tensor::from_vec(values, &[2, 128], &device).unwrap();

        let q_pt = loader.quantize_tensor(&tensor).unwrap();
        let rec_pt = loader.dequantize_tensor(&q_pt).unwrap();
        let err_pt = mse(&tensor, &rec_pt);

        let q_pc = loader.quantize_per_channel(&tensor, 0).unwrap();
        assert_eq!(q_pc.scales.len(), 2);
        assert_eq!(q_pc.zero_points.len(), 2);
        let rec_pc = loader.dequantize_per_channel(&q_pc).unwrap();
        let err_pc = mse(&tensor, &rec_pc);

        assert!(
            err_pc < err_pt,
            "per-channel MSE ({}) should be < per-tensor MSE ({})",
            err_pc,
            err_pt
        );
    }

    #[test]
    fn test_quantize_dtype_preservation() {
        let device = Device::Cpu;
        let loader = test_loader();

        let tensor = Tensor::randn(0.0_f32, 1.0_f32, (4, 5, 6), &device).unwrap();
        let original_dtype = tensor.dtype();
        let original_shape = tensor.dims().to_vec();

        let q = loader.quantize_tensor(&tensor).unwrap();
        assert_eq!(q.original_dtype, original_dtype);
        assert_eq!(q.original_shape, original_shape);
        assert_eq!(q.data.dtype(), DType::U8);
        assert_eq!(q.data.dims(), original_shape.as_slice());

        // Per-channel: same shape/dtype metadata, plus axis preserved.
        let q_pc = loader.quantize_per_channel(&tensor, 1).unwrap();
        assert_eq!(q_pc.original_dtype, original_dtype);
        assert_eq!(q_pc.original_shape, original_shape);
        assert_eq!(q_pc.axis, 1);
        assert_eq!(q_pc.scales.len(), original_shape[1]);
        assert_eq!(q_pc.zero_points.len(), original_shape[1]);
        assert_eq!(q_pc.data.dtype(), DType::U8);
        assert_eq!(q_pc.data.dims(), original_shape.as_slice());
    }

    #[test]
    fn test_dequantize_dtype_returns_to_original() {
        let device = Device::Cpu;
        let loader = test_loader();

        let tensor = Tensor::randn(0.0_f32, 1.0_f32, (32, 32), &device).unwrap();
        assert_eq!(tensor.dtype(), DType::F32);

        let q = loader.quantize_tensor(&tensor).unwrap();
        let recovered = loader.dequantize_tensor(&q).unwrap();
        assert_eq!(recovered.dtype(), DType::F32);
        assert_eq!(recovered.dims(), tensor.dims());

        let q_pc = loader.quantize_per_channel(&tensor, 0).unwrap();
        let recovered_pc = loader.dequantize_per_channel(&q_pc).unwrap();
        assert_eq!(recovered_pc.dtype(), DType::F32);
        assert_eq!(recovered_pc.dims(), tensor.dims());
    }
}
