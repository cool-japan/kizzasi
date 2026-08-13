//! Batch processing for efficient tokenization of multiple signals
//!
//! Provides batch encode/decode operations that can leverage
//! vectorization and parallelization for better performance.

use crate::error::{TokenizerError, TokenizerResult};
use crate::SignalTokenizer;
use scirs2_core::ndarray::{s, Array1, Array2};

/// Trait for batch tokenization operations
pub trait BatchTokenizer: SignalTokenizer {
    /// Encode multiple signals in batch
    ///
    /// Input: batch of signals [batch_size, signal_length]
    /// Output: batch of encodings [batch_size, embed_dim]
    ///
    /// Dispatches internally to the `rayon`-backed parallel path when the
    /// `parallel` feature is enabled (see
    /// [`BatchTokenizer::encode_batch_parallel`], still callable directly
    /// too), so callers get the speedup without needing to know about a
    /// separate method name.
    #[cfg(feature = "parallel")]
    fn encode_batch(&self, signals: &Array2<f32>) -> TokenizerResult<Array2<f32>>
    where
        Self: Sync,
    {
        self.encode_batch_parallel(signals)
    }

    /// Encode multiple signals in batch (see the `feature = "parallel"`
    /// version's docs above).
    #[cfg(not(feature = "parallel"))]
    fn encode_batch(&self, signals: &Array2<f32>) -> TokenizerResult<Array2<f32>> {
        let batch_size = signals.shape()[0];
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            // `SignalTokenizer::encode` takes `&Array1<f32>`, so a row view
            // still needs to be materialized into an owned array here.
            let signal = signals.row(i).to_owned();
            let encoded = self.encode(&signal)?;
            results.push(encoded);
        }

        // Stack results into batch
        batch_from_vec(results)
    }

    /// Decode multiple token sequences in batch
    ///
    /// Input: batch of tokens [batch_size, embed_dim]
    /// Output: batch of signals [batch_size, signal_length]
    ///
    /// Dispatches internally to the `rayon`-backed parallel path when the
    /// `parallel` feature is enabled (see
    /// [`BatchTokenizer::decode_batch_parallel`]).
    #[cfg(feature = "parallel")]
    fn decode_batch(&self, tokens: &Array2<f32>) -> TokenizerResult<Array2<f32>>
    where
        Self: Sync,
    {
        self.decode_batch_parallel(tokens)
    }

    /// Decode multiple token sequences in batch (see the `feature =
    /// "parallel"` version's docs above).
    #[cfg(not(feature = "parallel"))]
    fn decode_batch(&self, tokens: &Array2<f32>) -> TokenizerResult<Array2<f32>> {
        let batch_size = tokens.shape()[0];
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let token_seq = tokens.row(i).to_owned();
            let decoded = self.decode(&token_seq)?;
            results.push(decoded);
        }

        // Stack results into batch
        batch_from_vec(results)
    }

    /// Encode batch with padding to handle variable length signals
    ///
    /// Pads/truncates all signals to the specified target length before encoding.
    ///
    /// # Arguments
    /// * `signals` - Variable-length signals to encode
    /// * `target_len` - Target length to pad/truncate to (should match tokenizer's expected input dim)
    fn encode_batch_padded_to(
        &self,
        signals: &[Array1<f32>],
        target_len: usize,
    ) -> TokenizerResult<Array2<f32>> {
        if signals.is_empty() {
            return Err(TokenizerError::InvalidConfig("Empty batch".into()));
        }

        let mut results = Vec::with_capacity(signals.len());

        for signal in signals {
            // Pad or truncate to target length
            let mut padded = Array1::zeros(target_len);
            let copy_len = signal.len().min(target_len);
            padded
                .slice_mut(s![..copy_len])
                .assign(&signal.slice(s![..copy_len]));

            let encoded = self.encode(&padded)?;
            results.push(encoded);
        }

        // Stack results into batch
        batch_from_vec(results)
    }

    /// Process batch in parallel using multiple threads
    ///
    /// This can provide significant speedup for large batches
    #[cfg(feature = "parallel")]
    fn encode_batch_parallel(&self, signals: &Array2<f32>) -> TokenizerResult<Array2<f32>>
    where
        Self: Sync,
    {
        use rayon::prelude::*;

        let batch_size = signals.shape()[0];
        let results: Result<Vec<_>, _> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let signal = signals.row(i).to_owned();
                self.encode(&signal)
            })
            .collect();

        let results = results?;
        batch_from_vec(results)
    }

    /// Decode batch in parallel
    #[cfg(feature = "parallel")]
    fn decode_batch_parallel(&self, tokens: &Array2<f32>) -> TokenizerResult<Array2<f32>>
    where
        Self: Sync,
    {
        use rayon::prelude::*;

        let batch_size = tokens.shape()[0];
        let results: Result<Vec<_>, _> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let token_seq = tokens.row(i).to_owned();
                self.decode(&token_seq)
            })
            .collect();

        let results = results?;
        batch_from_vec(results)
    }
}

/// Helper function to stack Array1 vectors into Array2 batch
fn batch_from_vec(arrays: Vec<Array1<f32>>) -> TokenizerResult<Array2<f32>> {
    if arrays.is_empty() {
        return Err(TokenizerError::InvalidConfig("Empty batch".into()));
    }

    let batch_size = arrays.len();
    let elem_len = arrays[0].len();

    // Verify all arrays have same length
    for arr in arrays.iter() {
        if arr.len() != elem_len {
            return Err(TokenizerError::dim_mismatch(
                elem_len,
                arr.len(),
                "dimension validation",
            ));
        }
    }

    let mut batch = Array2::zeros((batch_size, elem_len));
    for (i, arr) in arrays.iter().enumerate() {
        batch.row_mut(i).assign(arr);
    }

    Ok(batch)
}

/// Streaming tokenizer for processing long sequences in chunks
pub struct StreamingTokenizer<T: SignalTokenizer> {
    tokenizer: T,
    chunk_size: usize,
    overlap: usize,
}

impl<T: SignalTokenizer> StreamingTokenizer<T> {
    /// Create a new streaming tokenizer
    ///
    /// # Arguments
    /// * `tokenizer` - The underlying tokenizer
    /// * `chunk_size` - Size of each chunk
    /// * `overlap` - Overlap between chunks (for continuity)
    pub fn new(tokenizer: T, chunk_size: usize, overlap: usize) -> TokenizerResult<Self> {
        if overlap >= chunk_size {
            return Err(TokenizerError::InvalidConfig(
                "Overlap must be less than chunk_size".into(),
            ));
        }

        Ok(Self {
            tokenizer,
            chunk_size,
            overlap,
        })
    }

    /// Encode a long signal in chunks
    pub fn encode_streaming(&self, signal: &Array1<f32>) -> TokenizerResult<Vec<Array1<f32>>> {
        let mut chunks = Vec::new();
        let stride = self.chunk_size - self.overlap;

        let mut start = 0;
        while start < signal.len() {
            let end = (start + self.chunk_size).min(signal.len());
            let chunk = Array1::from_vec(
                signal
                    .iter()
                    .skip(start)
                    .take(end - start)
                    .cloned()
                    .collect(),
            );

            // Pad last chunk if needed
            let chunk = if chunk.len() < self.chunk_size {
                let mut padded = Array1::zeros(self.chunk_size);
                for (i, &val) in chunk.iter().enumerate() {
                    padded[i] = val;
                }
                padded
            } else {
                chunk
            };

            chunks.push(self.tokenizer.encode(&chunk)?);
            start += stride;
        }

        Ok(chunks)
    }

    /// Decode chunks and reconstruct signal with overlap handling
    pub fn decode_streaming(&self, chunks: &[Array1<f32>]) -> TokenizerResult<Array1<f32>> {
        if chunks.is_empty() {
            return Err(TokenizerError::InvalidConfig("No chunks to decode".into()));
        }

        let mut decoded_chunks = Vec::new();
        for chunk in chunks {
            decoded_chunks.push(self.tokenizer.decode(chunk)?);
        }

        // Reconstruct with overlap blending
        let stride = self.chunk_size - self.overlap;
        let total_len = (chunks.len() - 1) * stride + self.chunk_size;
        let mut result = Array1::<f32>::zeros(total_len);
        let mut weight = Array1::<f32>::zeros(total_len);

        for (i, chunk) in decoded_chunks.iter().enumerate() {
            let start = i * stride;
            for (j, &val) in chunk.iter().enumerate() {
                if start + j < total_len {
                    result[start + j] += val;
                    weight[start + j] += 1.0;
                }
            }
        }

        // Average overlapping regions
        for i in 0..total_len {
            if weight[i] > 0.0 {
                result[i] /= weight[i];
            }
        }

        Ok(result)
    }

    /// Get the underlying tokenizer
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    /// Get chunk size
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get overlap size
    pub fn overlap(&self) -> usize {
        self.overlap
    }
}

// Implement BatchTokenizer for all SignalTokenizer types
impl<T: SignalTokenizer> BatchTokenizer for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ContinuousTokenizer;

    #[test]
    fn test_batch_from_vec() {
        let arrays = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![4.0, 5.0, 6.0]),
            Array1::from_vec(vec![7.0, 8.0, 9.0]),
        ];

        let batch = batch_from_vec(arrays).unwrap();
        assert_eq!(batch.shape(), &[3, 3]);
        assert_eq!(batch[[0, 0]], 1.0);
        assert_eq!(batch[[2, 2]], 9.0);
    }

    #[test]
    fn test_batch_from_vec_dimension_mismatch() {
        let arrays = vec![
            Array1::from_vec(vec![1.0, 2.0]),
            Array1::from_vec(vec![3.0, 4.0, 5.0]), // Wrong length
        ];

        assert!(batch_from_vec(arrays).is_err());
    }

    #[test]
    fn test_encode_batch() {
        let tokenizer = ContinuousTokenizer::new(4, 8);

        let signals = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);

        let encoded = tokenizer.encode_batch(&signals).unwrap();
        assert_eq!(encoded.shape(), &[3, 8]);
    }

    /// Regression: with the `parallel` feature enabled, the default
    /// `encode_batch`/`decode_batch` must dispatch to the same rayon-backed
    /// implementation as `encode_batch_parallel`/`decode_batch_parallel`
    /// (previously two unrelated code paths — this crate's own default
    /// `encode_batch` never parallelized no matter how the feature was
    /// set), and must produce identical results either way.
    #[test]
    #[cfg(feature = "parallel")]
    fn test_encode_decode_batch_matches_parallel_variant() {
        let tokenizer = ContinuousTokenizer::new(4, 8);
        let signals = Array2::from_shape_fn((5, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);

        let via_default = tokenizer.encode_batch(&signals).unwrap();
        let via_parallel = tokenizer.encode_batch_parallel(&signals).unwrap();
        assert_eq!(via_default, via_parallel);

        let tokens = Array2::from_shape_fn((5, 8), |(i, j)| (i * 8 + j) as f32 * 0.1);
        let decoded_default = tokenizer.decode_batch(&tokens).unwrap();
        let decoded_parallel = tokenizer.decode_batch_parallel(&tokens).unwrap();
        assert_eq!(decoded_default, decoded_parallel);
    }

    #[test]
    fn test_decode_batch() {
        let tokenizer = ContinuousTokenizer::new(4, 8);

        let tokens = Array2::from_shape_fn((3, 8), |(i, j)| (i * 8 + j) as f32 * 0.1);

        let decoded = tokenizer.decode_batch(&tokens).unwrap();
        assert_eq!(decoded.shape(), &[3, 4]);
    }

    #[test]
    fn test_encode_decode_batch_roundtrip() {
        let tokenizer = ContinuousTokenizer::new(10, 20);

        let signals = Array2::from_shape_fn((5, 10), |(i, j)| ((i * 10 + j) as f32 * 0.05).sin());

        let encoded = tokenizer.encode_batch(&signals).unwrap();
        let decoded = tokenizer.decode_batch(&encoded).unwrap();

        assert_eq!(decoded.shape(), signals.shape());
    }

    #[test]
    fn test_encode_batch_padded() {
        let tokenizer = ContinuousTokenizer::new(10, 20);

        let signals = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![4.0, 5.0, 6.0, 7.0, 8.0]),
            Array1::from_vec(vec![9.0, 10.0]),
        ];

        // Pad to input dimension (10)
        let encoded = tokenizer.encode_batch_padded_to(&signals, 10).unwrap();
        assert_eq!(encoded.shape(), &[3, 20]);
    }

    #[test]
    fn test_streaming_tokenizer() {
        let tokenizer = ContinuousTokenizer::new(8, 16);
        let streaming = StreamingTokenizer::new(tokenizer, 8, 2).unwrap();

        // Create a signal longer than chunk_size
        let signal = Array1::from_vec((0..20).map(|i| (i as f32 * 0.1).sin()).collect());

        let chunks = streaming.encode_streaming(&signal).unwrap();
        assert!(chunks.len() > 1); // Should be split into chunks

        let reconstructed = streaming.decode_streaming(&chunks).unwrap();
        // Allow some length difference due to padding
        assert!(reconstructed.len() >= signal.len());
    }

    #[test]
    fn test_streaming_tokenizer_overlap_validation() {
        let tokenizer = ContinuousTokenizer::new(8, 16);

        // Overlap >= chunk_size should fail
        assert!(StreamingTokenizer::new(tokenizer.clone(), 8, 8).is_err());
        assert!(StreamingTokenizer::new(tokenizer, 8, 9).is_err());
    }

    #[test]
    fn test_streaming_short_signal() {
        let tokenizer = ContinuousTokenizer::new(16, 32);
        let streaming = StreamingTokenizer::new(tokenizer, 16, 4).unwrap();

        // Signal shorter than chunk_size
        let signal = Array1::from_vec((0..10).map(|i| i as f32).collect());

        let chunks = streaming.encode_streaming(&signal).unwrap();
        assert_eq!(chunks.len(), 1); // Should be a single chunk

        let reconstructed = streaming.decode_streaming(&chunks).unwrap();
        assert_eq!(reconstructed.len(), 16); // Padded to chunk_size
    }
}
