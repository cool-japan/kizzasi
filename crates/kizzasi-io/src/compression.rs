//! Signal compression and decompression utilities
//!
//! This module provides various compression methods optimized for sensor data,
//! including lossless and lossy compression techniques.

use crate::error::{IoError, IoResult};
use std::collections::HashMap;

/// Compression method for signal data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMethod {
    /// No compression
    None,
    /// Run-Length Encoding (lossless, good for sparse signals)
    RLE,
    /// Delta encoding (store differences between samples)
    Delta,
    /// Delta + RLE combination
    DeltaRLE,
    /// Quantization (lossy, reduce bit depth)
    Quantize { bits: u8 },
    /// Differential Pulse Code Modulation
    DPCM { predictor_order: usize },
}

/// Compressed signal data
#[derive(Debug, Clone)]
pub struct CompressedSignal {
    /// Compression method used
    pub method: CompressionMethod,
    /// Compressed data bytes
    pub data: Vec<u8>,
    /// Original length
    pub original_length: usize,
    /// Metadata for decompression
    pub metadata: CompressionMetadata,
}

/// Metadata needed for decompression
#[derive(Debug, Clone)]
pub struct CompressionMetadata {
    /// Original sample rate
    pub sample_rate: Option<f32>,
    /// Original min value (for quantization)
    pub min_value: f32,
    /// Original max value (for quantization)
    pub max_value: f32,
    /// First sample (for delta encoding)
    pub first_sample: f32,
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

impl Default for CompressionMetadata {
    fn default() -> Self {
        Self {
            sample_rate: None,
            min_value: 0.0,
            max_value: 0.0,
            first_sample: 0.0,
            custom: HashMap::new(),
        }
    }
}

/// Signal compressor
pub struct SignalCompressor {
    method: CompressionMethod,
}

impl SignalCompressor {
    /// Create new compressor with specified method
    pub fn new(method: CompressionMethod) -> Self {
        Self { method }
    }

    /// Compress signal data
    pub fn compress(&self, signal: &[f32]) -> IoResult<CompressedSignal> {
        if signal.is_empty() {
            return Err(IoError::InvalidConfig(
                "Cannot compress empty signal".to_string(),
            ));
        }

        let min_value = signal.iter().copied().fold(f32::INFINITY, f32::min);
        let max_value = signal.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let metadata = CompressionMetadata {
            first_sample: signal[0],
            min_value,
            max_value,
            ..Default::default()
        };

        let data = match self.method {
            CompressionMethod::None => self.compress_none(signal)?,
            CompressionMethod::RLE => self.compress_rle(signal)?,
            CompressionMethod::Delta => self.compress_delta(signal)?,
            CompressionMethod::DeltaRLE => self.compress_delta_rle(signal)?,
            CompressionMethod::Quantize { bits } => {
                self.compress_quantize(signal, bits, metadata.min_value, metadata.max_value)?
            }
            CompressionMethod::DPCM { predictor_order } => {
                self.compress_dpcm(signal, predictor_order)?
            }
        };

        Ok(CompressedSignal {
            method: self.method,
            data,
            original_length: signal.len(),
            metadata,
        })
    }

    /// Decompress signal data
    pub fn decompress(&self, compressed: &CompressedSignal) -> IoResult<Vec<f32>> {
        match compressed.method {
            CompressionMethod::None => {
                self.decompress_none(&compressed.data, compressed.original_length)
            }
            CompressionMethod::RLE => self.decompress_rle(&compressed.data),
            CompressionMethod::Delta => {
                self.decompress_delta(&compressed.data, compressed.metadata.first_sample)
            }
            CompressionMethod::DeltaRLE => {
                self.decompress_delta_rle(&compressed.data, compressed.metadata.first_sample)
            }
            CompressionMethod::Quantize { bits } => self.decompress_quantize(
                &compressed.data,
                bits,
                compressed.metadata.min_value,
                compressed.metadata.max_value,
                compressed.original_length,
            ),
            CompressionMethod::DPCM { predictor_order } => self.decompress_dpcm(
                &compressed.data,
                predictor_order,
                compressed.metadata.first_sample,
            ),
        }
    }

    /// Get compression ratio
    pub fn compression_ratio(&self, original: &[f32], compressed: &CompressedSignal) -> f32 {
        let original_bytes = std::mem::size_of_val(original);
        let compressed_bytes = compressed.data.len();
        original_bytes as f32 / compressed_bytes as f32
    }

    // === Compression Methods ===

    fn compress_none(&self, signal: &[f32]) -> IoResult<Vec<u8>> {
        let mut data = Vec::with_capacity(std::mem::size_of_val(signal));
        for &sample in signal {
            data.extend_from_slice(&sample.to_le_bytes());
        }
        Ok(data)
    }

    fn compress_rle(&self, signal: &[f32]) -> IoResult<Vec<u8>> {
        let mut data = Vec::new();
        let mut i = 0;

        while i < signal.len() {
            let value = signal[i];
            let mut count = 1u16;

            // Count consecutive equal values
            while i + (count as usize) < signal.len()
                && signal[i + (count as usize)] == value
                && count < u16::MAX
            {
                count += 1;
            }

            // Store count and value
            data.extend_from_slice(&count.to_le_bytes());
            data.extend_from_slice(&value.to_le_bytes());

            i += count as usize;
        }

        Ok(data)
    }

    fn compress_delta(&self, signal: &[f32]) -> IoResult<Vec<u8>> {
        let mut data = Vec::with_capacity(std::mem::size_of_val(signal));

        for i in 1..signal.len() {
            let delta = signal[i] - signal[i - 1];
            data.extend_from_slice(&delta.to_le_bytes());
        }

        Ok(data)
    }

    fn compress_delta_rle(&self, signal: &[f32]) -> IoResult<Vec<u8>> {
        // First compute deltas
        let mut deltas = Vec::with_capacity(signal.len() - 1);
        for i in 1..signal.len() {
            deltas.push(signal[i] - signal[i - 1]);
        }

        // Then apply RLE to deltas
        self.compress_rle(&deltas)
    }

    fn compress_quantize(
        &self,
        signal: &[f32],
        bits: u8,
        min_val: f32,
        max_val: f32,
    ) -> IoResult<Vec<u8>> {
        if !quantize_bits_valid(bits) {
            return Err(IoError::InvalidConfig(
                "Quantization bits must be 1-16".to_string(),
            ));
        }

        let levels = (1u32 << bits) - 1;
        let range = max_val - min_val;
        // A degenerate (zero-width) range means every sample equals
        // min_val == max_val. Normalize to 0 instead of dividing by ~0, but
        // still emit one quantized value per input sample: a previous
        // version short-circuited to a single output byte here, which
        // silently dropped every sample but the first on decompress (see
        // `decompress_quantize`, which trusts `original_length`).
        let degenerate_range = range.abs() < 1e-10;

        let mut data = Vec::with_capacity(signal.len() * if bits <= 8 { 1 } else { 2 });

        for &sample in signal {
            let normalized = if degenerate_range {
                0.0
            } else {
                ((sample - min_val) / range).clamp(0.0, 1.0)
            };
            let quantized = (normalized * levels as f32).round() as u16;

            if bits <= 8 {
                data.push(quantized as u8);
            } else {
                data.extend_from_slice(&quantized.to_le_bytes());
            }
        }

        Ok(data)
    }

    fn compress_dpcm(&self, signal: &[f32], order: usize) -> IoResult<Vec<u8>> {
        if order == 0 || order >= signal.len() {
            return Err(IoError::InvalidConfig(
                "Invalid predictor order".to_string(),
            ));
        }

        let mut data = Vec::new();

        // Store initial samples
        for &sample in signal.iter().take(order) {
            data.extend_from_slice(&sample.to_le_bytes());
        }

        // Predict and store residuals
        for i in order..signal.len() {
            let predicted = self.linear_predict(&signal[i - order..i]);
            let residual = signal[i] - predicted;
            data.extend_from_slice(&residual.to_le_bytes());
        }

        Ok(data)
    }

    fn linear_predict(&self, history: &[f32]) -> f32 {
        // Simple linear prediction using average of previous samples
        if history.is_empty() {
            return 0.0;
        }
        history.iter().sum::<f32>() / history.len() as f32
    }

    // === Decompression Methods ===

    fn decompress_none(&self, data: &[u8], length: usize) -> IoResult<Vec<f32>> {
        let mut signal = Vec::with_capacity(length);

        for chunk in data.chunks_exact(4) {
            let bytes: [u8; 4] = chunk
                .try_into()
                .map_err(|_| IoError::ParseError("Invalid float data".to_string()))?;
            signal.push(f32::from_le_bytes(bytes));
        }

        Ok(signal)
    }

    fn decompress_rle(&self, data: &[u8]) -> IoResult<Vec<f32>> {
        let mut signal = Vec::new();
        let mut i = 0;

        while i + 6 <= data.len() {
            // Both slices are exactly 2 and 4 bytes long by construction of
            // the ranges above, so `try_into()` cannot actually fail here --
            // but `map_err` (rather than `expect`) keeps this function
            // panic-free in non-test library code even if that invariant is
            // ever violated by a future edit.
            let count_bytes: [u8; 2] = data[i..i + 2].try_into().map_err(|_| {
                IoError::ParseError("RLE decode: count chunk must be 2 bytes".to_string())
            })?;
            let count = u16::from_le_bytes(count_bytes);

            let value_bytes: [u8; 4] = data[i + 2..i + 6].try_into().map_err(|_| {
                IoError::ParseError("RLE decode: value chunk must be 4 bytes".to_string())
            })?;
            let value = f32::from_le_bytes(value_bytes);

            for _ in 0..count {
                signal.push(value);
            }

            i += 6;
        }

        Ok(signal)
    }

    fn decompress_delta(&self, data: &[u8], first_sample: f32) -> IoResult<Vec<f32>> {
        // Track the running reconstructed value directly instead of
        // re-reading it back out via `Vec::last()` (which would need an
        // `expect`/`unwrap` to justify why the vector, seeded with
        // `first_sample`, can never be empty).
        let mut signal = Vec::with_capacity(data.len() / 4 + 1);
        signal.push(first_sample);
        let mut current = first_sample;

        for chunk in data.chunks_exact(4) {
            let bytes: [u8; 4] = chunk
                .try_into()
                .map_err(|_| IoError::ParseError("Invalid delta data".to_string()))?;
            let delta = f32::from_le_bytes(bytes);
            current += delta;
            signal.push(current);
        }

        Ok(signal)
    }

    fn decompress_delta_rle(&self, data: &[u8], first_sample: f32) -> IoResult<Vec<f32>> {
        let deltas = self.decompress_rle(data)?;
        let mut signal = Vec::with_capacity(deltas.len() + 1);
        signal.push(first_sample);
        let mut current = first_sample;

        for delta in deltas {
            current += delta;
            signal.push(current);
        }

        Ok(signal)
    }

    fn decompress_quantize(
        &self,
        data: &[u8],
        bits: u8,
        min_val: f32,
        max_val: f32,
        length: usize,
    ) -> IoResult<Vec<f32>> {
        if !quantize_bits_valid(bits) {
            return Err(IoError::ParseError(
                "Quantization bits must be 1-16".to_string(),
            ));
        }

        let levels = (1u32 << bits) - 1;
        let range = max_val - min_val;
        let mut signal = Vec::with_capacity(length);

        if bits <= 8 {
            for &byte in data {
                let normalized = byte as f32 / levels as f32;
                let value = min_val + normalized * range;
                signal.push(value);
            }
        } else {
            for chunk in data.chunks_exact(2) {
                let bytes: [u8; 2] = chunk.try_into().map_err(|_| {
                    IoError::ParseError("Quantization decode: chunk must be 2 bytes".to_string())
                })?;
                let quantized = u16::from_le_bytes(bytes);
                let normalized = quantized as f32 / levels as f32;
                let value = min_val + normalized * range;
                signal.push(value);
            }
        }

        Ok(signal)
    }

    fn decompress_dpcm(&self, data: &[u8], order: usize, first_sample: f32) -> IoResult<Vec<f32>> {
        if order == 0 {
            return Err(IoError::ParseError(
                "DPCM decode: predictor order must be > 0".to_string(),
            ));
        }

        // `order * 4` is attacker/corruption-controlled (it comes straight
        // from the caller-supplied `CompressedSignal`), so it must be
        // checked for overflow and validated against the actual data length
        // before being used as a slice bound. Both loops below must use the
        // SAME validated `init_bytes` -- a previous version validated only
        // the first loop and let the second slice `data[init_bytes..]`
        // straight through, panicking whenever `init_bytes > data.len()`.
        let init_bytes = order.checked_mul(4).ok_or_else(|| {
            IoError::ParseError("DPCM decode: predictor order overflow".to_string())
        })?;

        if init_bytes > data.len() {
            return Err(IoError::ParseError(format!(
                "DPCM decode: predictor order {} requires {} initial bytes but only {} are available",
                order,
                init_bytes,
                data.len()
            )));
        }

        let mut signal = Vec::new();

        // Read initial samples
        for chunk in data[..init_bytes].chunks_exact(4) {
            let bytes: [u8; 4] = chunk.try_into().map_err(|_| {
                IoError::ParseError("DPCM decode: chunk must be 4 bytes".to_string())
            })?;
            signal.push(f32::from_le_bytes(bytes));
        }

        if signal.is_empty() {
            signal.push(first_sample);
        }

        // Reconstruct from residuals
        for chunk in data[init_bytes..].chunks_exact(4) {
            let bytes: [u8; 4] = chunk.try_into().map_err(|_| {
                IoError::ParseError("DPCM decode: chunk must be 4 bytes".to_string())
            })?;
            let residual = f32::from_le_bytes(bytes);
            let predicted = self.linear_predict(&signal[signal.len().saturating_sub(order)..]);
            signal.push(predicted + residual);
        }

        Ok(signal)
    }
}

/// Shared bit-depth validation for `CompressionMethod::Quantize`, used by
/// both the compress and decompress paths so they cannot silently drift out
/// of sync with each other again.
fn quantize_bits_valid(bits: u8) -> bool {
    (1..=16).contains(&bits)
}

/// Adaptive compressor that selects best method
pub struct AdaptiveCompressor {
    methods: Vec<CompressionMethod>,
}

impl AdaptiveCompressor {
    /// Create adaptive compressor with default methods
    pub fn new() -> Self {
        Self {
            methods: vec![
                CompressionMethod::Delta,
                CompressionMethod::DeltaRLE,
                CompressionMethod::Quantize { bits: 8 },
                CompressionMethod::DPCM { predictor_order: 4 },
            ],
        }
    }

    /// Compress using best method
    pub fn compress(&self, signal: &[f32]) -> IoResult<CompressedSignal> {
        let mut best_compressed: Option<CompressedSignal> = None;
        let mut best_size = usize::MAX;

        for &method in &self.methods {
            let compressor = SignalCompressor::new(method);
            if let Ok(compressed) = compressor.compress(signal) {
                if compressed.data.len() < best_size {
                    best_size = compressed.data.len();
                    best_compressed = Some(compressed);
                }
            }
        }

        best_compressed
            .ok_or_else(|| IoError::SignalError("All compression methods failed".to_string()))
    }

    /// Decompress signal
    pub fn decompress(&self, compressed: &CompressedSignal) -> IoResult<Vec<f32>> {
        let compressor = SignalCompressor::new(compressed.method);
        compressor.decompress(compressed)
    }
}

impl Default for AdaptiveCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_none() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let compressor = SignalCompressor::new(CompressionMethod::None);

        let compressed = compressor.compress(&signal).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(signal, decompressed);
    }

    #[test]
    fn test_compress_delta() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let compressor = SignalCompressor::new(CompressionMethod::Delta);

        let compressed = compressor.compress(&signal).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        for (a, b) in signal.iter().zip(decompressed.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compress_rle() {
        let signal = vec![1.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let compressor = SignalCompressor::new(CompressionMethod::RLE);

        let compressed = compressor.compress(&signal).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(signal, decompressed);

        // RLE should compress better than raw
        let raw_size = signal.len() * 4;
        assert!(compressed.data.len() < raw_size);
    }

    #[test]
    fn test_compress_quantize() {
        let signal = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let compressor = SignalCompressor::new(CompressionMethod::Quantize { bits: 8 });

        let compressed = compressor.compress(&signal).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        // Allow small error due to quantization
        for (a, b) in signal.iter().zip(decompressed.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn test_adaptive_compressor() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let compressor = AdaptiveCompressor::new();

        let compressed = compressor.compress(&signal).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(signal.len(), decompressed.len());
    }

    #[test]
    fn test_compression_ratio() {
        let signal = vec![1.0; 100]; // Highly compressible
        let compressor = SignalCompressor::new(CompressionMethod::RLE);

        let compressed = compressor.compress(&signal).unwrap();
        let ratio = compressor.compression_ratio(&signal, &compressed);

        assert!(ratio > 10.0); // Should compress very well
    }

    // === Regression tests: decompress_dpcm out-of-bounds panic (critical) ===

    #[test]
    fn test_decompress_dpcm_truncated_data_returns_error_not_panic() {
        let compressor = SignalCompressor::new(CompressionMethod::DPCM { predictor_order: 8 });
        let compressed = CompressedSignal {
            method: CompressionMethod::DPCM { predictor_order: 8 },
            // Far too short for order=8 (needs 32 bytes just for the seed
            // samples): must error, not panic on `data[init_bytes..]`.
            data: vec![0u8; 4],
            original_length: 100,
            metadata: CompressionMetadata::default(),
        };
        let result = compressor.decompress(&compressed);
        assert!(result.is_err(), "expected an error, got {:?}", result);
    }

    #[test]
    fn test_decompress_dpcm_huge_order_does_not_overflow_or_panic() {
        let compressor = SignalCompressor::new(CompressionMethod::DPCM {
            predictor_order: usize::MAX / 2,
        });
        let compressed = CompressedSignal {
            method: CompressionMethod::DPCM {
                predictor_order: usize::MAX / 2,
            },
            data: vec![0u8; 16],
            original_length: 10,
            metadata: CompressionMetadata::default(),
        };
        // `order * 4` would overflow usize here; must be a clean error.
        let result = compressor.decompress(&compressed);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_dpcm_order_zero_is_rejected() {
        let compressor = SignalCompressor::new(CompressionMethod::DPCM { predictor_order: 0 });
        let compressed = CompressedSignal {
            method: CompressionMethod::DPCM { predictor_order: 0 },
            data: vec![0u8; 16],
            original_length: 4,
            metadata: CompressionMetadata::default(),
        };
        assert!(compressor.decompress(&compressed).is_err());
    }

    #[test]
    fn test_dpcm_round_trip_still_works() {
        let signal = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
        let compressor = SignalCompressor::new(CompressionMethod::DPCM { predictor_order: 2 });
        let compressed = compressor.compress(&signal).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), signal.len());
        for (a, b) in signal.iter().zip(decompressed.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    // === Regression tests: constant-signal Quantize data loss (high) ===

    #[test]
    fn test_compress_quantize_constant_signal_preserves_length() {
        let signal = vec![3.0; 10_000];
        let compressor = SignalCompressor::new(CompressionMethod::Quantize { bits: 8 });
        let compressed = compressor.compress(&signal).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(
            decompressed.len(),
            signal.len(),
            "constant-signal round trip must not silently truncate to a single sample"
        );
        for &value in &decompressed {
            assert!((value - 3.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_adaptive_compressor_constant_signal_round_trips() {
        // AdaptiveCompressor picks the smallest encoding, which used to be
        // the lossy 1-byte Quantize path for any near-constant signal.
        let signal = vec![-2.5; 500];
        let compressor = AdaptiveCompressor::new();
        let compressed = compressor.compress(&signal).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), signal.len());
    }

    // === Regression tests: decompress_quantize bit validation (high) ===

    #[test]
    fn test_decompress_quantize_rejects_zero_bits() {
        let compressor = SignalCompressor::new(CompressionMethod::Quantize { bits: 0 });
        let compressed = CompressedSignal {
            method: CompressionMethod::Quantize { bits: 0 },
            data: vec![1, 2, 3],
            original_length: 3,
            metadata: CompressionMetadata {
                min_value: 0.0,
                max_value: 1.0,
                ..Default::default()
            },
        };
        let result = compressor.decompress(&compressed);
        assert!(result.is_err(), "bits=0 must error, not divide by zero");
    }

    #[test]
    fn test_decompress_quantize_rejects_excessive_bits() {
        let compressor = SignalCompressor::new(CompressionMethod::Quantize { bits: 200 });
        let compressed = CompressedSignal {
            method: CompressionMethod::Quantize { bits: 200 },
            data: vec![1, 2, 3, 4],
            original_length: 2,
            metadata: CompressionMetadata {
                min_value: 0.0,
                max_value: 1.0,
                ..Default::default()
            },
        };
        // bits=200 would overflow `1u32 << bits`; must error cleanly.
        let result = compressor.decompress(&compressed);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantize_bits_validation_matches_compress_and_decompress() {
        // compress() already validated bits=0/>16; decompress() must now
        // agree instead of silently producing inf/NaN.
        let compressor = SignalCompressor::new(CompressionMethod::Quantize { bits: 0 });
        assert!(compressor.compress(&[1.0, 2.0]).is_err());
    }
}
