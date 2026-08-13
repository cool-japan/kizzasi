//! State compression for memory efficiency
//!
//! This module provides compression algorithms to reduce the memory footprint
//! of hidden states during inference. This is especially useful for:
//! - Long sequence generation
//! - Resource-constrained environments
//! - Distributed inference with state transfer
//!
//! ## Compression Methods
//!
//! 1. **Quantization**: Reduce precision of state values
//! 2. **Sparse encoding**: Store only non-zero values
//! 3. **Low-rank approximation**: SVD-based compression
//! 4. **Dictionary encoding**: Store frequently occurring patterns

use crate::error::{InferenceError, InferenceResult};
use kizzasi_core::HiddenState;
use scirs2_core::ndarray::Array2;

/// Compression method for hidden states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMethod {
    /// No compression
    None,
    /// Quantize to 8-bit integers
    Quantize8Bit,
    /// Quantize to 4-bit integers
    Quantize4Bit,
    /// Sparse encoding (store only non-zero values above threshold)
    Sparse,
    /// Combination of quantization and sparsity
    QuantizedSparse,
}

/// Compressed representation of a hidden state
#[derive(Debug, Clone)]
pub struct CompressedState {
    /// Compression method used
    method: CompressionMethod,
    /// Compressed data
    data: Vec<u8>,
    /// Original shape
    shape: Vec<usize>,
    /// Scaling factor (for quantization)
    scale: f32,
    /// Zero point (for quantization)
    zero_point: i32,
    /// Sparsity metadata (indices of non-zero elements)
    sparse_indices: Option<Vec<usize>>,
    /// Set when every quantized element (or, for `QuantizedSparse`, every
    /// surviving above-threshold element) shares the same value, making a
    /// scale-based quantization degenerate (`scale` would be `0.0`).
    ///
    /// When `Some`, decompression reconstructs the state directly from this
    /// value instead of decoding `data`, so the round-trip is exact instead
    /// of landing on `0.0` via a `scale == 0.0` / `zero_point == i32::MIN`
    /// computation.
    degenerate_value: Option<f32>,
}

impl CompressedState {
    /// Get the compression ratio achieved
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.shape.iter().product::<usize>() * std::mem::size_of::<f32>();
        let compressed_size = self.data.len()
            + self
                .sparse_indices
                .as_ref()
                .map(|v| v.len() * std::mem::size_of::<usize>())
                .unwrap_or(0)
            + self
                .degenerate_value
                .map(|_| std::mem::size_of::<f32>())
                .unwrap_or(0);

        if compressed_size == 0 {
            // Nothing at all was stored: only possible for a zero-element
            // state, where "original / compressed" (0 / 0) is not a
            // meaningful ratio.
            return if original_size == 0 {
                1.0
            } else {
                f32::INFINITY
            };
        }

        original_size as f32 / compressed_size as f32
    }

    /// Get compression method
    pub fn method(&self) -> CompressionMethod {
        self.method
    }
}

/// State compressor with configurable compression method
pub struct StateCompressor {
    method: CompressionMethod,
    /// Sparsity threshold (values below this are treated as zero)
    sparsity_threshold: f32,
}

impl StateCompressor {
    /// Create a new state compressor
    pub fn new(method: CompressionMethod) -> Self {
        Self {
            method,
            sparsity_threshold: 1e-4,
        }
    }

    /// Set sparsity threshold
    pub fn with_sparsity_threshold(mut self, threshold: f32) -> Self {
        self.sparsity_threshold = threshold;
        self
    }

    /// Compress a hidden state
    pub fn compress(&self, state: &HiddenState) -> InferenceResult<CompressedState> {
        match self.method {
            CompressionMethod::None => self.compress_none(state),
            CompressionMethod::Quantize8Bit => self.compress_quantize_8bit(state),
            CompressionMethod::Quantize4Bit => self.compress_quantize_4bit(state),
            CompressionMethod::Sparse => self.compress_sparse(state),
            CompressionMethod::QuantizedSparse => self.compress_quantized_sparse(state),
        }
    }

    /// Decompress a compressed state
    pub fn decompress(&self, compressed: &CompressedState) -> InferenceResult<HiddenState> {
        match compressed.method {
            CompressionMethod::None => self.decompress_none(compressed),
            CompressionMethod::Quantize8Bit => self.decompress_quantize_8bit(compressed),
            CompressionMethod::Quantize4Bit => self.decompress_quantize_4bit(compressed),
            CompressionMethod::Sparse => self.decompress_sparse(compressed),
            CompressionMethod::QuantizedSparse => self.decompress_quantized_sparse(compressed),
        }
    }

    /// No compression - just copy
    fn compress_none(&self, state: &HiddenState) -> InferenceResult<CompressedState> {
        let data_vec: Vec<f32> = state.state().iter().copied().collect();
        let data_bytes: Vec<u8> = data_vec.iter().flat_map(|&f| f.to_le_bytes()).collect();

        let shape_vec: Vec<usize> = state.state().shape().to_vec();

        Ok(CompressedState {
            method: CompressionMethod::None,
            data: data_bytes,
            shape: shape_vec,
            scale: 1.0,
            zero_point: 0,
            sparse_indices: None,
            degenerate_value: None,
        })
    }

    fn decompress_none(&self, compressed: &CompressedState) -> InferenceResult<HiddenState> {
        let floats: Vec<f32> = compressed
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let data = Array2::from_shape_vec((compressed.shape[0], compressed.shape[1]), floats)
            .map_err(|e| InferenceError::ForwardError(e.to_string()))?;

        let mut hidden = HiddenState::new(compressed.shape[0], compressed.shape[1]);
        hidden.update(data);
        Ok(hidden)
    }

    /// 8-bit quantization
    ///
    /// A constant-valued (or empty) state makes a `(max - min) / 255` scale
    /// degenerate (`0.0`, which would drive `zero_point` to `i32::MIN` and
    /// every decoded element to `0.0`). Both cases are stored as an exact
    /// [`CompressedState::degenerate_value`] instead.
    fn compress_quantize_8bit(&self, state: &HiddenState) -> InferenceResult<CompressedState> {
        let shape_vec: Vec<usize> = state.state().shape().to_vec();

        if state.state().is_empty() {
            return Ok(CompressedState {
                method: CompressionMethod::Quantize8Bit,
                data: Vec::new(),
                shape: shape_vec,
                scale: 1.0,
                zero_point: 0,
                sparse_indices: None,
                degenerate_value: None,
            });
        }

        let min_val = state.state().iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = state
            .state()
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        if max_val == min_val {
            return Ok(CompressedState {
                method: CompressionMethod::Quantize8Bit,
                data: Vec::new(),
                shape: shape_vec,
                scale: 1.0,
                zero_point: 0,
                sparse_indices: None,
                degenerate_value: Some(max_val),
            });
        }

        let scale = (max_val - min_val) / 255.0;
        let zero_point = (-min_val / scale).round() as i32;

        let quantized: Vec<u8> = state
            .state()
            .iter()
            .map(|&v| {
                let scaled = (v / scale + zero_point as f32).round();
                scaled.clamp(0.0, 255.0) as u8
            })
            .collect();

        Ok(CompressedState {
            method: CompressionMethod::Quantize8Bit,
            data: quantized,
            shape: shape_vec,
            scale,
            zero_point,
            sparse_indices: None,
            degenerate_value: None,
        })
    }

    fn decompress_quantize_8bit(
        &self,
        compressed: &CompressedState,
    ) -> InferenceResult<HiddenState> {
        if let Some(value) = compressed.degenerate_value {
            return Self::dense_from_constant(&compressed.shape, value);
        }

        let dequantized: Vec<f32> = compressed
            .data
            .iter()
            .map(|&q| (q as f32 - compressed.zero_point as f32) * compressed.scale)
            .collect();

        let data = Array2::from_shape_vec((compressed.shape[0], compressed.shape[1]), dequantized)
            .map_err(|e| InferenceError::ForwardError(e.to_string()))?;

        let mut hidden = HiddenState::new(compressed.shape[0], compressed.shape[1]);
        hidden.update(data);
        Ok(hidden)
    }

    /// 4-bit quantization (2 values per byte)
    ///
    /// See [`StateCompressor::compress_quantize_8bit`]: constant-valued and
    /// empty states are handled the same way here, for the same reason.
    fn compress_quantize_4bit(&self, state: &HiddenState) -> InferenceResult<CompressedState> {
        let shape_vec: Vec<usize> = state.state().shape().to_vec();

        if state.state().is_empty() {
            return Ok(CompressedState {
                method: CompressionMethod::Quantize4Bit,
                data: Vec::new(),
                shape: shape_vec,
                scale: 1.0,
                zero_point: 0,
                sparse_indices: None,
                degenerate_value: None,
            });
        }

        let min_val = state.state().iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = state
            .state()
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        if max_val == min_val {
            return Ok(CompressedState {
                method: CompressionMethod::Quantize4Bit,
                data: Vec::new(),
                shape: shape_vec,
                scale: 1.0,
                zero_point: 0,
                sparse_indices: None,
                degenerate_value: Some(max_val),
            });
        }

        let scale = (max_val - min_val) / 15.0;
        let zero_point = (-min_val / scale).round() as i32;

        let mut quantized = Vec::new();
        let mut iter = state.state().iter();

        while let Some(&v1) = iter.next() {
            let q1 = ((v1 / scale + zero_point as f32).round().clamp(0.0, 15.0) as u8) & 0x0F;
            let q2 = if let Some(&v2) = iter.next() {
                ((v2 / scale + zero_point as f32).round().clamp(0.0, 15.0) as u8) & 0x0F
            } else {
                0
            };
            quantized.push((q1 << 4) | q2);
        }

        Ok(CompressedState {
            method: CompressionMethod::Quantize4Bit,
            data: quantized,
            shape: shape_vec,
            scale,
            zero_point,
            sparse_indices: None,
            degenerate_value: None,
        })
    }

    fn decompress_quantize_4bit(
        &self,
        compressed: &CompressedState,
    ) -> InferenceResult<HiddenState> {
        if let Some(value) = compressed.degenerate_value {
            return Self::dense_from_constant(&compressed.shape, value);
        }

        let total_elements = compressed.shape.iter().product();
        let mut dequantized = Vec::with_capacity(total_elements);

        for &byte in &compressed.data {
            let q1 = (byte >> 4) & 0x0F;
            let q2 = byte & 0x0F;

            dequantized.push((q1 as f32 - compressed.zero_point as f32) * compressed.scale);
            if dequantized.len() < total_elements {
                dequantized.push((q2 as f32 - compressed.zero_point as f32) * compressed.scale);
            }
        }

        let data = Array2::from_shape_vec((compressed.shape[0], compressed.shape[1]), dequantized)
            .map_err(|e| InferenceError::ForwardError(e.to_string()))?;

        let mut hidden = HiddenState::new(compressed.shape[0], compressed.shape[1]);
        hidden.update(data);
        Ok(hidden)
    }

    /// Sparse encoding
    fn compress_sparse(&self, state: &HiddenState) -> InferenceResult<CompressedState> {
        let mut values = Vec::new();
        let mut indices = Vec::new();

        for (i, &v) in state.state().iter().enumerate() {
            if v.abs() > self.sparsity_threshold {
                values.push(v);
                indices.push(i);
            }
        }

        let data_bytes: Vec<u8> = values.iter().flat_map(|&f| f.to_le_bytes()).collect();

        let shape_vec: Vec<usize> = state.state().shape().to_vec();

        Ok(CompressedState {
            method: CompressionMethod::Sparse,
            data: data_bytes,
            shape: shape_vec,
            scale: 1.0,
            zero_point: 0,
            sparse_indices: Some(indices),
            degenerate_value: None,
        })
    }

    fn decompress_sparse(&self, compressed: &CompressedState) -> InferenceResult<HiddenState> {
        let values: Vec<f32> = compressed
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let indices = compressed
            .sparse_indices
            .as_ref()
            .ok_or(InferenceError::ForwardError(
                "Missing sparse indices".to_string(),
            ))?;

        let total_elements: usize = compressed.shape.iter().product();
        let mut dense = vec![0.0f32; total_elements];
        for (&idx, &val) in indices.iter().zip(values.iter()) {
            if idx < dense.len() {
                dense[idx] = val;
            }
        }

        let data = Array2::from_shape_vec((compressed.shape[0], compressed.shape[1]), dense)
            .map_err(|e| InferenceError::ForwardError(e.to_string()))?;

        let mut hidden = HiddenState::new(compressed.shape[0], compressed.shape[1]);
        hidden.update(data);
        Ok(hidden)
    }

    /// Combined quantized sparse encoding
    ///
    /// The *surviving* (above-threshold) values can themselves be constant
    /// even when the full dense state is not (e.g. several equal spikes on a
    /// zero background) — the same degeneracy as
    /// [`StateCompressor::compress_quantize_8bit`], handled the same way.
    fn compress_quantized_sparse(&self, state: &HiddenState) -> InferenceResult<CompressedState> {
        let mut values = Vec::new();
        let mut indices = Vec::new();

        for (i, &v) in state.state().iter().enumerate() {
            if v.abs() > self.sparsity_threshold {
                values.push(v);
                indices.push(i);
            }
        }

        let shape_vec: Vec<usize> = state.state().shape().to_vec();

        if values.is_empty() {
            return Ok(CompressedState {
                method: CompressionMethod::QuantizedSparse,
                data: Vec::new(),
                shape: shape_vec,
                scale: 1.0,
                zero_point: 0,
                sparse_indices: Some(indices),
                degenerate_value: None,
            });
        }

        let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        if max_val == min_val {
            return Ok(CompressedState {
                method: CompressionMethod::QuantizedSparse,
                data: Vec::new(),
                shape: shape_vec,
                scale: 1.0,
                zero_point: 0,
                sparse_indices: Some(indices),
                degenerate_value: Some(max_val),
            });
        }

        let scale = (max_val - min_val) / 255.0;
        let zero_point = (-min_val / scale).round() as i32;

        let quantized: Vec<u8> = values
            .iter()
            .map(|&v| {
                let scaled = (v / scale + zero_point as f32).round();
                scaled.clamp(0.0, 255.0) as u8
            })
            .collect();

        Ok(CompressedState {
            method: CompressionMethod::QuantizedSparse,
            data: quantized,
            shape: shape_vec,
            scale,
            zero_point,
            sparse_indices: Some(indices),
            degenerate_value: None,
        })
    }

    fn decompress_quantized_sparse(
        &self,
        compressed: &CompressedState,
    ) -> InferenceResult<HiddenState> {
        let indices = compressed
            .sparse_indices
            .as_ref()
            .ok_or(InferenceError::ForwardError(
                "Missing sparse indices".to_string(),
            ))?;

        let total_elements: usize = compressed.shape.iter().product();
        let mut dense = vec![0.0f32; total_elements];

        if let Some(value) = compressed.degenerate_value {
            for &idx in indices {
                if idx < dense.len() {
                    dense[idx] = value;
                }
            }
        } else if !compressed.data.is_empty() {
            for (&idx, &q) in indices.iter().zip(compressed.data.iter()) {
                if idx < dense.len() {
                    dense[idx] = (q as f32 - compressed.zero_point as f32) * compressed.scale;
                }
            }
        }

        let data = Array2::from_shape_vec((compressed.shape[0], compressed.shape[1]), dense)
            .map_err(|e| InferenceError::ForwardError(e.to_string()))?;

        let mut hidden = HiddenState::new(compressed.shape[0], compressed.shape[1]);
        hidden.update(data);
        Ok(hidden)
    }

    /// Build a dense `HiddenState` of `shape` where every element is `value`.
    ///
    /// Shared by the degenerate-value decode path of every quantization
    /// method. `shape` is always 2-D in practice (it always originates from
    /// an `Array2`'s own `.shape()`), but this reports a clear error for a
    /// malformed shape rather than indexing into it unchecked.
    fn dense_from_constant(shape: &[usize], value: f32) -> InferenceResult<HiddenState> {
        let &[rows, cols] = shape else {
            return Err(InferenceError::ForwardError(format!(
                "expected a 2-D shape, got {shape:?}"
            )));
        };
        let data = Array2::from_shape_vec((rows, cols), vec![value; rows * cols])
            .map_err(|e| InferenceError::ForwardError(e.to_string()))?;
        let mut hidden = HiddenState::new(rows, cols);
        hidden.update(data);
        Ok(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::random::{rng, RngExt};

    /// Helper: build a `HiddenState` from an `Array2<f32>` using the public API.
    fn hidden_from_array(arr: Array2<f32>) -> HiddenState {
        let shape = arr.shape();
        let mut state = HiddenState::new(shape[0], shape[1]);
        state.update(arr);
        state
    }

    /// Helper: compute mean-squared error between two equally-shaped states.
    fn mse(a: &HiddenState, b: &HiddenState) -> f32 {
        let lhs = a.state();
        let rhs = b.state();
        assert_eq!(lhs.shape(), rhs.shape(), "shape mismatch in mse helper");
        let n = lhs.len().max(1) as f32;
        lhs.iter()
            .zip(rhs.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            / n
    }

    /// Helper: produce a uniform [-1, 1] random `Array2<f32>` using SciRS2 RNG.
    fn random_signed(rows: usize, cols: usize) -> Array2<f32> {
        let mut generator = rng();
        Array2::from_shape_fn((rows, cols), |_| generator.random::<f32>() * 2.0 - 1.0)
    }

    // ---- Roundtrip / quantization fidelity ----------------------------------

    // Test 1: compress/decompress a `[32, 64]` dense state and require
    // MSE < 1e-2 (8-bit quantization), with shape preserved.
    #[test]
    fn test_compress_decompress_roundtrip_dense() {
        let original = hidden_from_array(random_signed(32, 64));

        let compressor = StateCompressor::new(CompressionMethod::Quantize8Bit);
        let compressed = compressor.compress(&original).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed.state().shape(), original.state().shape());
        let error = mse(&original, &decompressed);
        assert!(error < 1e-2, "8-bit roundtrip MSE {} exceeded 1e-2", error);
    }

    // Test 2: a `[64, 128]` state where 90% of values are zero and 10% are
    // random in `[-1, 1]`. Sparse encoding must compress the value payload
    // below 25% of the dense element count, and roundtrip MSE must be < 1e-2.
    #[test]
    fn test_compress_decompress_sparse_high_ratio() {
        let rows = 64;
        let cols = 128;
        let total = rows * cols;

        let mut generator = rng();
        let arr = Array2::from_shape_fn((rows, cols), |_| {
            if generator.random::<f32>() < 0.9 {
                0.0
            } else {
                generator.random::<f32>() * 2.0 - 1.0
            }
        });
        let state = hidden_from_array(arr);

        // threshold is small (1e-4 default) so any non-zero we set survives,
        // and the zeros are filtered out.
        let compressor = StateCompressor::new(CompressionMethod::Sparse);
        let compressed = compressor.compress(&state).unwrap();

        // Sparse method stores 4-byte f32s, so payload element count is
        // data.len() / 4. Compare against the dense element count.
        let stored_elems = compressed.data.len() / std::mem::size_of::<f32>();
        let limit = total / 4; // 25%
        assert!(
            stored_elems < limit,
            "sparse stored_elems {} should be < {} (25% of {})",
            stored_elems,
            limit,
            total
        );

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.state().shape(), state.state().shape());
        let error = mse(&state, &decompressed);
        assert!(error < 1e-2, "sparse roundtrip MSE {} exceeded 1e-2", error);
    }

    // Test 3: run 10 compress/decompress iterations on a random `[16, 32]`
    // state and verify the cumulative drift (final-vs-original MSE) is bounded.
    #[test]
    fn test_compress_decompress_multistep_drift() {
        let original = hidden_from_array(random_signed(16, 32));
        let compressor = StateCompressor::new(CompressionMethod::Quantize8Bit);

        let mut current = original.clone();
        for _ in 0..10 {
            let compressed = compressor.compress(&current).unwrap();
            current = compressor.decompress(&compressed).unwrap();
        }

        // After the first dequantization, quantization is idempotent because
        // the dequantized values land exactly on lattice points; the only
        // drift comes from min/max moving slightly between rounds. We allow
        // a loose 1e-3 bound on cumulative MSE.
        let drift = mse(&original, &current);
        assert!(
            drift < 1e-3,
            "cumulative 10-step drift MSE {} exceeded 1e-3",
            drift
        );
    }

    // Test 4: after 8-bit compression, the recorded `scale` must be > 0, the
    // `zero_point` must fit inside the u8 range, and the shape must be
    // preserved exactly.
    #[test]
    fn test_compress_preserves_quantization_params() {
        let original = hidden_from_array(random_signed(8, 16));
        let compressor = StateCompressor::new(CompressionMethod::Quantize8Bit);
        let compressed = compressor.compress(&original).unwrap();

        assert!(
            compressed.scale > 0.0,
            "scale must be positive, got {}",
            compressed.scale
        );
        assert!(
            (0..=255).contains(&compressed.zero_point),
            "zero_point {} must lie in [0, 255]",
            compressed.zero_point
        );
        assert_eq!(compressed.shape, vec![8, 16]);
        assert_eq!(compressed.method(), CompressionMethod::Quantize8Bit);
    }

    // Test 5: a `[0, 0]` state must round-trip cleanly through every
    // compression method, including 8-bit/4-bit quantization — which used to
    // compute `scale` from the min/max of an empty iterator and produce NaN.
    #[test]
    fn test_compress_empty_state() {
        let original = hidden_from_array(Array2::<f32>::zeros((0, 0)));

        for method in [
            CompressionMethod::None,
            CompressionMethod::Quantize8Bit,
            CompressionMethod::Quantize4Bit,
            CompressionMethod::Sparse,
            CompressionMethod::QuantizedSparse,
        ] {
            let compressor = StateCompressor::new(method);
            let compressed = compressor
                .compress(&original)
                .unwrap_or_else(|e| panic!("{:?} compress failed on empty state: {:?}", method, e));
            assert_eq!(compressed.shape, vec![0, 0]);

            let decompressed = compressor.decompress(&compressed).unwrap_or_else(|e| {
                panic!("{:?} decompress failed on empty state: {:?}", method, e)
            });
            assert_eq!(decompressed.state().shape(), &[0, 0]);
            assert_eq!(decompressed.state().len(), 0);
        }
    }

    // Test 6: a `[1, 1]` state with a single value round-trips losslessly
    // through every method, including 8-bit/4-bit quantization (previously
    // degenerate whenever `min == max`, i.e. always for a single element).
    #[test]
    fn test_compress_single_element() {
        let mut arr = Array2::<f32>::zeros((1, 1));
        arr[[0, 0]] = 0.42;
        let original = hidden_from_array(arr);

        for method in [
            CompressionMethod::None,
            CompressionMethod::Quantize8Bit,
            CompressionMethod::Quantize4Bit,
        ] {
            let compressor = StateCompressor::new(method);
            let compressed = compressor.compress(&original).unwrap();
            let decompressed = compressor.decompress(&compressed).unwrap();

            assert_eq!(decompressed.state().shape(), &[1, 1]);
            assert!(
                (decompressed.state()[[0, 0]] - 0.42).abs() < 1e-6,
                "{:?}: 1x1 roundtrip drifted: got {}",
                method,
                decompressed.state()[[0, 0]]
            );
        }
    }

    /// Regression: `QuantizedSparse`'s *surviving* (above-threshold) values
    /// can be constant even when the full dense state is not (e.g. equal
    /// spikes on a zero background) — the same `scale == 0.0` degeneracy as
    /// the dense quantizers, previously unhandled here.
    #[test]
    fn test_compress_quantized_sparse_degenerate_survivors() {
        let mut arr = Array2::<f32>::zeros((2, 4));
        arr[[0, 0]] = 0.9;
        arr[[0, 2]] = 0.9;
        arr[[1, 1]] = 0.9;
        let state = hidden_from_array(arr);

        let compressor = StateCompressor::new(CompressionMethod::QuantizedSparse);
        let compressed = compressor.compress(&state).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        for (got, expected) in decompressed.state().iter().zip(state.state().iter()) {
            assert!(
                (got - expected).abs() < 1e-6,
                "QuantizedSparse degenerate-survivor round-trip drifted: got {} expected {}",
                got,
                expected
            );
        }
    }

    // Test 7: an all-zero state must round-trip back to all-zero (within a
    // tight tolerance). Cover both Sparse and None paths.
    #[test]
    fn test_compress_all_zeros() {
        let original = hidden_from_array(Array2::<f32>::zeros((16, 16)));

        // Sparse should compress to an empty payload because no value clears
        // the threshold.
        let sparse = StateCompressor::new(CompressionMethod::Sparse);
        let compressed = sparse.compress(&original).unwrap();
        assert!(compressed.data.is_empty());
        let decompressed = sparse.decompress(&compressed).unwrap();
        assert_eq!(decompressed.state().shape(), &[16, 16]);
        for v in decompressed.state().iter() {
            assert!((*v).abs() < 1e-6);
        }

        // None compression should also reproduce zeros exactly.
        let none = StateCompressor::new(CompressionMethod::None);
        let nc = none.compress(&original).unwrap();
        let nd = none.decompress(&nc).unwrap();
        for v in nd.state().iter() {
            assert_eq!(*v, 0.0);
        }
    }

    // Test 8: a state where every entry is 0.5. The 8-bit/4-bit quantizers
    // used to collapse to a degenerate `scale == 0.0` here (and
    // `zero_point` would saturate to `i32::MIN`), silently decoding to
    // all-zeros. They must now round-trip exactly, with a finite,
    // better-than-1x compression ratio (not the `inf`/near-zero a
    // `scale == 0.0` bug would produce).
    #[test]
    fn test_compress_all_identical() {
        let original = hidden_from_array(Array2::<f32>::from_elem((16, 16), 0.5));

        for method in [
            CompressionMethod::Quantize8Bit,
            CompressionMethod::Quantize4Bit,
        ] {
            let compressor = StateCompressor::new(method);
            let compressed = compressor.compress(&original).unwrap();
            let decompressed = compressor.decompress(&compressed).unwrap();

            assert_eq!(decompressed.state().shape(), &[16, 16]);
            for v in decompressed.state().iter() {
                assert!(
                    (v - 0.5).abs() < 1e-6,
                    "{:?}: all-identical drifted: got {}",
                    method,
                    v
                );
            }
            let ratio = compressed.compression_ratio();
            assert!(
                ratio.is_finite() && ratio > 1.0,
                "{:?}: compression_ratio must be finite and > 1.0, got {}",
                method,
                ratio
            );
        }

        // The lossless `None` path remains available too.
        let compressor = StateCompressor::new(CompressionMethod::None);
        let compressed = compressor.compress(&original).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        for v in decompressed.state().iter() {
            assert!((v - 0.5).abs() < 1e-6, "all-identical drifted: got {}", v);
        }
    }

    // Test 9: three different aspect ratios — square, wide, tall — should all
    // round-trip through 8-bit compression with their shape preserved.
    #[test]
    fn test_decompress_preserves_shape() {
        let shapes = [(8usize, 8usize), (1, 64), (64, 1)];
        let compressor = StateCompressor::new(CompressionMethod::Quantize8Bit);

        for (rows, cols) in shapes {
            let state = hidden_from_array(random_signed(rows, cols));
            let compressed = compressor.compress(&state).unwrap();
            let decompressed = compressor.decompress(&compressed).unwrap();
            assert_eq!(
                decompressed.state().shape(),
                &[rows, cols],
                "shape mismatch for input ({}, {})",
                rows,
                cols
            );
            assert_eq!(compressed.shape, vec![rows, cols]);
        }
    }

    // Test 10: sparse thresholding — with a threshold of 0.1, only magnitudes
    // strictly above 0.1 should be retained in the sparse payload. Values
    // equal to or below the threshold (in absolute value) must be dropped.
    #[test]
    fn test_compress_with_sparse_threshold_boundary() {
        // Layout: four entries, two on each side of the boundary.
        //   0.05  -> dropped (below)
        //   0.10  -> dropped (equal — the impl uses strict `>` )
        //   0.11  -> kept    (just above)
        //   0.50  -> kept
        let mut arr = Array2::<f32>::zeros((1, 4));
        arr[[0, 0]] = 0.05;
        arr[[0, 1]] = 0.10;
        arr[[0, 2]] = 0.11;
        arr[[0, 3]] = 0.50;

        let state = hidden_from_array(arr);
        let compressor =
            StateCompressor::new(CompressionMethod::Sparse).with_sparsity_threshold(0.1);
        let compressed = compressor.compress(&state).unwrap();

        let indices = compressed
            .sparse_indices
            .as_ref()
            .expect("sparse indices must be present");
        assert_eq!(
            indices,
            &vec![2usize, 3usize],
            "only 0.11 and 0.50 should clear the strict-> threshold"
        );

        let stored_elems = compressed.data.len() / std::mem::size_of::<f32>();
        assert_eq!(stored_elems, 2);

        let decompressed = compressor.decompress(&compressed).unwrap();
        // Dropped entries are reconstructed as zero.
        assert!((decompressed.state()[[0, 0]]).abs() < 1e-6);
        assert!((decompressed.state()[[0, 1]]).abs() < 1e-6);
        assert!((decompressed.state()[[0, 2]] - 0.11).abs() < 1e-6);
        assert!((decompressed.state()[[0, 3]] - 0.50).abs() < 1e-6);
    }
}
