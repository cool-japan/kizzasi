//! Variable-length Sequence Handling
//!
//! Provides utilities for efficiently handling sequences of variable length:
//! - Padding and masking
//! - Packed sequence representation
//! - Efficient batch processing
//! - Length-aware operations

use crate::error::{CoreError, CoreResult};
use scirs2_core::ndarray::{Array1, Array2, Array3};

/// Padding strategy for variable-length sequences
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// Pad sequences to the right (default)
    Right,
    /// Pad sequences to the left
    Left,
    /// No padding (all sequences must have same length)
    None,
}

/// Sequence mask for variable-length batches
///
/// Tracks which positions in a batch are valid vs. padding
#[derive(Debug, Clone)]
pub struct SequenceMask {
    /// Boolean mask: true = valid position, false = padding
    /// Shape: (batch_size, seq_len)
    mask: Array2<bool>,
    /// Actual lengths of each sequence in the batch
    lengths: Array1<usize>,
    /// Maximum sequence length in the batch
    max_len: usize,
}

impl SequenceMask {
    /// Create a new sequence mask from lengths
    pub fn from_lengths(lengths: &[usize]) -> CoreResult<Self> {
        if lengths.is_empty() {
            return Err(CoreError::InvalidConfig(
                "Cannot create mask from empty lengths".to_string(),
            ));
        }

        let batch_size = lengths.len();
        let max_len = *lengths.iter().max().unwrap();

        if max_len == 0 {
            return Err(CoreError::InvalidConfig(
                "Max length must be greater than 0".to_string(),
            ));
        }

        // Create mask array
        let mut mask = Array2::from_elem((batch_size, max_len), false);

        for (i, &length) in lengths.iter().enumerate() {
            if length > max_len {
                return Err(CoreError::InvalidConfig(format!(
                    "Length {} exceeds max_len {}",
                    length, max_len
                )));
            }
            for j in 0..length {
                mask[[i, j]] = true;
            }
        }

        let lengths_array = Array1::from_vec(lengths.to_vec());

        Ok(Self {
            mask,
            lengths: lengths_array,
            max_len,
        })
    }

    /// Get the boolean mask
    pub fn mask(&self) -> &Array2<bool> {
        &self.mask
    }

    /// Get sequence lengths
    pub fn lengths(&self) -> &Array1<usize> {
        &self.lengths
    }

    /// Get maximum length
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.lengths.len()
    }

    /// Check if a position is valid (not padding)
    pub fn is_valid(&self, batch_idx: usize, seq_idx: usize) -> bool {
        if batch_idx >= self.batch_size() || seq_idx >= self.max_len {
            return false;
        }
        self.mask[[batch_idx, seq_idx]]
    }

    /// Count total number of valid (non-padding) positions
    pub fn count_valid(&self) -> usize {
        self.mask.iter().filter(|&&x| x).count()
    }
}

/// Packed sequence representation for efficient processing
///
/// Stores only the valid (non-padded) elements in a contiguous array
#[derive(Debug, Clone)]
pub struct PackedSequence {
    /// Packed data (only valid elements)
    /// Shape: (total_valid_elements, feature_dim)
    data: Array2<f32>,
    /// Batch indices for each element
    batch_indices: Array1<usize>,
    /// Sorted lengths (for efficient unpacking)
    sorted_lengths: Array1<usize>,
    /// Original batch size
    batch_size: usize,
    /// Feature dimension
    feature_dim: usize,
}

impl PackedSequence {
    /// Pack a batch of variable-length sequences
    ///
    /// Input shape: (batch_size, max_seq_len, feature_dim)
    /// Mask shape: (batch_size, max_seq_len)
    pub fn pack(sequences: &Array3<f32>, mask: &SequenceMask) -> CoreResult<Self> {
        let (batch_size, max_seq_len, feature_dim) = sequences.dim();

        if batch_size != mask.batch_size() {
            return Err(CoreError::DimensionMismatch {
                expected: mask.batch_size(),
                got: batch_size,
            });
        }

        if max_seq_len != mask.max_len() {
            return Err(CoreError::DimensionMismatch {
                expected: mask.max_len(),
                got: max_seq_len,
            });
        }

        let total_valid = mask.count_valid();

        // Allocate packed arrays
        let mut data = Array2::zeros((total_valid, feature_dim));
        let mut batch_indices = Array1::zeros(total_valid);

        // Pack data
        let mut idx = 0;
        for b in 0..batch_size {
            let length = mask.lengths()[b];
            for t in 0..length {
                // Copy features
                for f in 0..feature_dim {
                    data[[idx, f]] = sequences[[b, t, f]];
                }
                batch_indices[idx] = b;
                idx += 1;
            }
        }

        Ok(Self {
            data,
            batch_indices,
            sorted_lengths: mask.lengths().clone(),
            batch_size,
            feature_dim,
        })
    }

    /// Unpack back to padded batch format
    ///
    /// Output shape: (batch_size, max_seq_len, feature_dim)
    pub fn unpack(&self, padding_value: f32) -> CoreResult<Array3<f32>> {
        let max_len = *self.sorted_lengths.iter().max().unwrap();
        let mut output =
            Array3::from_elem((self.batch_size, max_len, self.feature_dim), padding_value);

        let mut idx = 0;
        for b in 0..self.batch_size {
            let length = self.sorted_lengths[b];
            for t in 0..length {
                for f in 0..self.feature_dim {
                    output[[b, t, f]] = self.data[[idx, f]];
                }
                idx += 1;
            }
        }

        Ok(output)
    }

    /// Get packed data
    pub fn data(&self) -> &Array2<f32> {
        &self.data
    }

    /// Get batch indices
    pub fn batch_indices(&self) -> &Array1<usize> {
        &self.batch_indices
    }

    /// Get total number of valid elements
    pub fn num_elements(&self) -> usize {
        self.data.nrows()
    }
}

/// Pad sequences to the same length
///
/// Input: Vec of sequences with shape (seq_len, feature_dim)
/// Output: Padded array with shape (batch_size, max_seq_len, feature_dim) and mask
pub fn pad_sequences(
    sequences: &[Array2<f32>],
    padding_value: f32,
    strategy: PaddingStrategy,
) -> CoreResult<(Array3<f32>, SequenceMask)> {
    if sequences.is_empty() {
        return Err(CoreError::InvalidConfig(
            "Cannot pad empty sequence list".to_string(),
        ));
    }

    let batch_size = sequences.len();
    let feature_dim = sequences[0].ncols();

    // Collect lengths and find max
    let lengths: Vec<usize> = sequences.iter().map(|s| s.nrows()).collect();
    let max_len = *lengths.iter().max().unwrap();

    // Check feature dimensions match
    for (i, seq) in sequences.iter().enumerate() {
        if seq.ncols() != feature_dim {
            return Err(CoreError::InvalidConfig(format!(
                "Feature dimension mismatch at index {}: expected {}, got {}",
                i,
                feature_dim,
                seq.ncols()
            )));
        }
    }

    // Create padded array
    let mut padded = Array3::from_elem((batch_size, max_len, feature_dim), padding_value);

    // Fill in sequences based on padding strategy
    for (b, seq) in sequences.iter().enumerate() {
        let seq_len = seq.nrows();

        match strategy {
            PaddingStrategy::Right => {
                // Pad on the right (default)
                for t in 0..seq_len {
                    for f in 0..feature_dim {
                        padded[[b, t, f]] = seq[[t, f]];
                    }
                }
            }
            PaddingStrategy::Left => {
                // Pad on the left
                let offset = max_len - seq_len;
                for t in 0..seq_len {
                    for f in 0..feature_dim {
                        padded[[b, offset + t, f]] = seq[[t, f]];
                    }
                }
            }
            PaddingStrategy::None => {
                if seq_len != max_len {
                    return Err(CoreError::InvalidConfig(format!(
                        "Sequence {} has length {} but max_len is {}. Use padding strategy.",
                        b, seq_len, max_len
                    )));
                }
                for t in 0..seq_len {
                    for f in 0..feature_dim {
                        padded[[b, t, f]] = seq[[t, f]];
                    }
                }
            }
        }
    }

    // Create mask
    let mask = SequenceMask::from_lengths(&lengths)?;

    Ok((padded, mask))
}

/// Apply mask to a tensor by zeroing out padding positions
pub fn apply_mask(tensor: &mut Array3<f32>, mask: &SequenceMask, mask_value: f32) {
    let (batch_size, seq_len, feature_dim) = tensor.dim();

    for b in 0..batch_size {
        for t in 0..seq_len {
            if !mask.is_valid(b, t) {
                for f in 0..feature_dim {
                    tensor[[b, t, f]] = mask_value;
                }
            }
        }
    }
}

/// Compute sequence-aware mean (ignoring padding)
pub fn masked_mean(tensor: &Array3<f32>, mask: &SequenceMask) -> CoreResult<Array2<f32>> {
    let (batch_size, seq_len, feature_dim) = tensor.dim();

    if batch_size != mask.batch_size() {
        return Err(CoreError::DimensionMismatch {
            expected: mask.batch_size(),
            got: batch_size,
        });
    }

    let mut result = Array2::zeros((batch_size, feature_dim));

    for b in 0..batch_size {
        let length = mask.lengths()[b] as f32;
        if length == 0.0 {
            continue;
        }

        for t in 0..seq_len {
            if mask.is_valid(b, t) {
                for f in 0..feature_dim {
                    result[[b, f]] += tensor[[b, t, f]] / length;
                }
            }
        }
    }

    Ok(result)
}

/// Compute sequence-aware sum (ignoring padding)
pub fn masked_sum(tensor: &Array3<f32>, mask: &SequenceMask) -> CoreResult<Array2<f32>> {
    let (batch_size, seq_len, feature_dim) = tensor.dim();

    if batch_size != mask.batch_size() {
        return Err(CoreError::DimensionMismatch {
            expected: mask.batch_size(),
            got: batch_size,
        });
    }

    let mut result = Array2::zeros((batch_size, feature_dim));

    for b in 0..batch_size {
        for t in 0..seq_len {
            if mask.is_valid(b, t) {
                for f in 0..feature_dim {
                    result[[b, f]] += tensor[[b, t, f]];
                }
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_mask() {
        let lengths = vec![3, 5, 2];
        let mask = SequenceMask::from_lengths(&lengths).unwrap();

        assert_eq!(mask.batch_size(), 3);
        assert_eq!(mask.max_len(), 5);
        assert_eq!(mask.count_valid(), 10); // 3 + 5 + 2

        // Check specific positions
        assert!(mask.is_valid(0, 0));
        assert!(mask.is_valid(0, 2));
        assert!(!mask.is_valid(0, 3));

        assert!(mask.is_valid(1, 4));
        assert!(!mask.is_valid(1, 5));

        assert!(mask.is_valid(2, 1));
        assert!(!mask.is_valid(2, 2));
    }

    #[test]
    fn test_pad_sequences() {
        let seq1 = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        let seq2 = Array2::from_shape_vec((4, 3), vec![2.0; 12]).unwrap();
        let seq3 = Array2::from_shape_vec((3, 3), vec![3.0; 9]).unwrap();

        let sequences = vec![seq1, seq2, seq3];
        let (padded, mask) = pad_sequences(&sequences, 0.0, PaddingStrategy::Right).unwrap();

        assert_eq!(padded.dim(), (3, 4, 3)); // batch_size=3, max_len=4, feature_dim=3
        assert_eq!(mask.max_len(), 4);
        assert_eq!(mask.lengths()[0], 2);
        assert_eq!(mask.lengths()[1], 4);
        assert_eq!(mask.lengths()[2], 3);

        // Check values
        assert_eq!(padded[[0, 0, 0]], 1.0);
        assert_eq!(padded[[0, 2, 0]], 0.0); // padding

        assert_eq!(padded[[1, 3, 0]], 2.0);
        assert_eq!(padded[[2, 2, 0]], 3.0);
    }

    #[test]
    fn test_packed_sequence() {
        let lengths = vec![2, 3, 1];
        let mask = SequenceMask::from_lengths(&lengths).unwrap();

        let mut sequences = Array3::zeros((3, 3, 2)); // batch=3, max_len=3, features=2
                                                      // Fill with test data
        for b in 0..3 {
            for t in 0..lengths[b] {
                for f in 0..2 {
                    sequences[[b, t, f]] = (b * 10 + t) as f32;
                }
            }
        }

        let packed = PackedSequence::pack(&sequences, &mask).unwrap();
        assert_eq!(packed.num_elements(), 6); // 2 + 3 + 1

        let unpacked = packed.unpack(0.0).unwrap();
        assert_eq!(unpacked.dim(), (3, 3, 2));

        // Check that valid positions match
        for b in 0..3 {
            for t in 0..lengths[b] {
                for f in 0..2 {
                    assert_eq!(sequences[[b, t, f]], unpacked[[b, t, f]]);
                }
            }
        }
    }

    #[test]
    fn test_masked_mean() {
        let lengths = vec![2, 3];
        let mask = SequenceMask::from_lengths(&lengths).unwrap();

        let mut sequences = Array3::zeros((2, 3, 2));
        // Batch 0: [[1, 1], [2, 2], [0, 0]] with length 2 -> mean = [1.5, 1.5]
        sequences[[0, 0, 0]] = 1.0;
        sequences[[0, 0, 1]] = 1.0;
        sequences[[0, 1, 0]] = 2.0;
        sequences[[0, 1, 1]] = 2.0;

        // Batch 1: [[3, 3], [4, 4], [5, 5]] with length 3 -> mean = [4, 4]
        sequences[[1, 0, 0]] = 3.0;
        sequences[[1, 0, 1]] = 3.0;
        sequences[[1, 1, 0]] = 4.0;
        sequences[[1, 1, 1]] = 4.0;
        sequences[[1, 2, 0]] = 5.0;
        sequences[[1, 2, 1]] = 5.0;

        let mean = masked_mean(&sequences, &mask).unwrap();

        assert!((mean[[0, 0]] - 1.5).abs() < 1e-6);
        assert!((mean[[0, 1]] - 1.5).abs() < 1e-6);
        assert!((mean[[1, 0]] - 4.0).abs() < 1e-6);
        assert!((mean[[1, 1]] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_mask() {
        let lengths = vec![2, 1];
        let mask = SequenceMask::from_lengths(&lengths).unwrap();

        let mut sequences = Array3::from_elem((2, 3, 2), 1.0);
        apply_mask(&mut sequences, &mask, 0.0);

        // Check that padding positions are zeroed
        assert_eq!(sequences[[0, 0, 0]], 1.0);
        assert_eq!(sequences[[0, 1, 0]], 1.0);
        assert_eq!(sequences[[0, 2, 0]], 0.0); // padding

        assert_eq!(sequences[[1, 0, 0]], 1.0);
        assert_eq!(sequences[[1, 1, 0]], 0.0); // padding
        assert_eq!(sequences[[1, 2, 0]], 0.0); // padding
    }
}
