//! Batched Inference Support
//!
//! Provides efficient batched inference for all model architectures.
//! Manages batch-aware state handling, padding, and dynamic batching.
//!
//! # Features
//!
//! - **Batch-aware state management**: Separate hidden states per batch item
//! - **Efficient padding**: Dynamic sequence length handling
//! - **Dynamic batching**: Adaptive batch size optimization
//! - **Memory efficient**: Shared computation when possible
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi_model::batch::BatchedModel;
//! use kizzasi_model::mamba::Mamba;
//!
//! let model = Mamba::new(config)?;
//! let batched = BatchedModel::new(model, batch_size);
//!
//! // Process batch of inputs
//! let outputs = batched.predict_batch(&inputs)?;
//! ```

use crate::error::{ModelError, ModelResult};
use crate::AutoregressiveModel;
use kizzasi_core::HiddenState;
use scirs2_core::ndarray::{s, Array2, Array3};

/// Batched wrapper for autoregressive models
///
/// Manages multiple sequences in parallel with independent states
pub struct BatchedModel<M: AutoregressiveModel> {
    /// Underlying model
    model: M,
    /// Batch size
    batch_size: usize,
    /// Hidden states for each batch item [batch_size, num_layers, state_shape]
    batch_states: Vec<Vec<HiddenState>>,
    /// Sequence lengths for each batch item
    sequence_lengths: Vec<usize>,
    /// Maximum sequence length in current batch
    max_length: usize,
}

impl<M: AutoregressiveModel> BatchedModel<M> {
    /// Create a new batched model wrapper
    ///
    /// # Arguments
    ///
    /// * `model` - The underlying autoregressive model
    /// * `batch_size` - Number of sequences to process in parallel
    pub fn new(model: M, batch_size: usize) -> ModelResult<Self> {
        if batch_size == 0 {
            return Err(ModelError::invalid_config("batch_size must be > 0"));
        }

        // Initialize states for each batch item
        // Get initial states from the model to ensure correct dimensions
        let template_states = model.get_states();

        // Clone template states for each batch item
        let batch_states = (0..batch_size).map(|_| template_states.clone()).collect();

        Ok(Self {
            model,
            batch_size,
            batch_states,
            sequence_lengths: vec![0; batch_size],
            max_length: 0,
        })
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Predict for a batch of inputs
    ///
    /// # Arguments
    ///
    /// * `inputs` - Batch of inputs [batch_size, input_dim]
    ///
    /// # Returns
    ///
    /// Batch of predictions [batch_size, output_dim]
    pub fn predict_batch(&mut self, inputs: &Array2<f32>) -> ModelResult<Array2<f32>> {
        let (batch_size, _input_dim) = inputs.dim();

        if batch_size != self.batch_size {
            return Err(ModelError::dimension_mismatch(
                "batch predict",
                self.batch_size,
                batch_size,
            ));
        }

        let output_dim = self.model.hidden_dim();
        let mut outputs = Array2::zeros((batch_size, output_dim));

        // Process each item in the batch
        for batch_idx in 0..batch_size {
            let input = inputs.row(batch_idx).to_owned();

            // Set model state for this batch item
            self.model
                .set_states(self.batch_states[batch_idx].clone())?;

            // Predict using step method from SignalPredictor trait
            let output = self.model.step(&input)?;

            // Store output
            outputs.row_mut(batch_idx).assign(&output);

            // Update stored state for this batch item
            self.batch_states[batch_idx] = self.model.get_states();

            // Update sequence length
            self.sequence_lengths[batch_idx] += 1;
        }

        self.max_length = *self.sequence_lengths.iter().max().unwrap_or(&0);

        Ok(outputs)
    }

    /// Predict multiple steps for a batch
    ///
    /// # Arguments
    ///
    /// * `inputs` - Batch of input sequences [batch_size, seq_len, input_dim]
    ///
    /// # Returns
    ///
    /// Batch of prediction sequences [batch_size, seq_len, output_dim]
    pub fn predict_sequence_batch(&mut self, inputs: &Array3<f32>) -> ModelResult<Array3<f32>> {
        let (batch_size, seq_len, _input_dim) = inputs.dim();

        if batch_size != self.batch_size {
            return Err(ModelError::dimension_mismatch(
                "batch forward_sequence",
                self.batch_size,
                batch_size,
            ));
        }

        let output_dim = self.model.hidden_dim();
        let mut outputs = Array3::zeros((batch_size, seq_len, output_dim));

        // Process each timestep
        for t in 0..seq_len {
            // Extract inputs for this timestep
            let step_inputs = inputs.slice(s![.., t, ..]).to_owned();

            // Predict for all batch items
            let step_outputs = self.predict_batch(&step_inputs)?;

            // Store outputs
            outputs.slice_mut(s![.., t, ..]).assign(&step_outputs);
        }

        Ok(outputs)
    }

    /// Reset states for specific batch items
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices of batch items to reset
    pub fn reset_batch_items(&mut self, indices: &[usize]) -> ModelResult<()> {
        // Reset model and get fresh initial states
        self.model.reset();
        let template_states = self.model.get_states();

        for &idx in indices {
            if idx >= self.batch_size {
                return Err(ModelError::invalid_config(format!(
                    "batch index {} out of range for batch_size {}",
                    idx, self.batch_size
                )));
            }

            // Reset state for this batch item
            self.batch_states[idx] = template_states.clone();
            self.sequence_lengths[idx] = 0;
        }

        Ok(())
    }

    /// Reset all batch states
    pub fn reset_all(&mut self) {
        // Reset model and get fresh initial states
        self.model.reset();
        let template_states = self.model.get_states();

        self.batch_states = (0..self.batch_size)
            .map(|_| template_states.clone())
            .collect();

        self.sequence_lengths = vec![0; self.batch_size];
        self.max_length = 0;
    }

    /// Get sequence lengths for all batch items
    pub fn sequence_lengths(&self) -> &[usize] {
        &self.sequence_lengths
    }

    /// Get maximum sequence length in the batch
    pub fn max_sequence_length(&self) -> usize {
        self.max_length
    }

    /// Get states for a specific batch item
    pub fn get_batch_item_states(&self, batch_idx: usize) -> ModelResult<&Vec<HiddenState>> {
        if batch_idx >= self.batch_size {
            return Err(ModelError::invalid_config(format!(
                "batch index {} out of range for batch_size {}",
                batch_idx, self.batch_size
            )));
        }
        Ok(&self.batch_states[batch_idx])
    }

    /// Set states for a specific batch item
    pub fn set_batch_item_states(
        &mut self,
        batch_idx: usize,
        states: Vec<HiddenState>,
    ) -> ModelResult<()> {
        if batch_idx >= self.batch_size {
            return Err(ModelError::invalid_config(format!(
                "batch index {} out of range for batch_size {}",
                batch_idx, self.batch_size
            )));
        }

        if states.len() != self.model.num_layers() {
            return Err(ModelError::invalid_config(format!(
                "expected {} layer states, got {}",
                self.model.num_layers(),
                states.len()
            )));
        }

        self.batch_states[batch_idx] = states;
        Ok(())
    }

    /// Get reference to underlying model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get mutable reference to underlying model
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

/// Dynamic batching strategy
///
/// Automatically adjusts batch size based on memory constraints
/// and computational efficiency
pub struct DynamicBatcher<M: AutoregressiveModel> {
    /// Base model
    #[allow(dead_code)]
    model: M,
    /// Minimum batch size
    min_batch_size: usize,
    /// Maximum batch size
    max_batch_size: usize,
    /// Current optimal batch size
    current_batch_size: usize,
    /// Target latency in microseconds
    target_latency_us: u64,
}

impl<M: AutoregressiveModel> DynamicBatcher<M> {
    /// Create a new dynamic batcher
    ///
    /// # Arguments
    ///
    /// * `model` - The underlying model
    /// * `min_batch_size` - Minimum batch size
    /// * `max_batch_size` - Maximum batch size
    /// * `target_latency_us` - Target latency per batch in microseconds
    pub fn new(
        model: M,
        min_batch_size: usize,
        max_batch_size: usize,
        target_latency_us: u64,
    ) -> ModelResult<Self> {
        if min_batch_size == 0 {
            return Err(ModelError::invalid_config("min_batch_size must be > 0"));
        }
        if max_batch_size < min_batch_size {
            return Err(ModelError::invalid_config(
                "max_batch_size must be >= min_batch_size",
            ));
        }

        Ok(Self {
            model,
            min_batch_size,
            max_batch_size,
            current_batch_size: min_batch_size,
            target_latency_us,
        })
    }

    /// Get current optimal batch size
    pub fn current_batch_size(&self) -> usize {
        self.current_batch_size
    }

    /// Update batch size based on observed latency
    ///
    /// # Arguments
    ///
    /// * `observed_latency_us` - Observed latency for last batch
    pub fn update_batch_size(&mut self, observed_latency_us: u64) {
        // Simple adaptive strategy
        if observed_latency_us < self.target_latency_us {
            // We have headroom, can increase batch size
            self.current_batch_size = (self.current_batch_size + 1).min(self.max_batch_size);
        } else if observed_latency_us > self.target_latency_us * 2 {
            // Too slow, decrease batch size
            self.current_batch_size = (self.current_batch_size - 1).max(self.min_batch_size);
        }
        // Otherwise keep current batch size
    }

    /// Reset to minimum batch size
    pub fn reset(&mut self) {
        self.current_batch_size = self.min_batch_size;
    }
}

/// Batch padding utilities
pub mod padding {
    use super::*;

    /// Pad batch to maximum length
    ///
    /// # Arguments
    ///
    /// * `sequences` - Variable-length sequences
    /// * `pad_value` - Value to use for padding
    ///
    /// # Returns
    ///
    /// Padded array [batch_size, max_len, features] and original lengths
    pub fn pad_sequences(sequences: &[Array2<f32>], pad_value: f32) -> (Array3<f32>, Vec<usize>) {
        if sequences.is_empty() {
            return (Array3::zeros((0, 0, 0)), vec![]);
        }

        let batch_size = sequences.len();
        let max_len = sequences.iter().map(|s| s.nrows()).max().unwrap_or(0);
        let feature_dim = sequences[0].ncols();

        let mut padded = Array3::from_elem((batch_size, max_len, feature_dim), pad_value);
        let mut lengths = Vec::with_capacity(batch_size);

        for (batch_idx, seq) in sequences.iter().enumerate() {
            let seq_len = seq.nrows();
            lengths.push(seq_len);

            for t in 0..seq_len {
                for f in 0..feature_dim {
                    padded[[batch_idx, t, f]] = seq[[t, f]];
                }
            }
        }

        (padded, lengths)
    }

    /// Remove padding from batch
    ///
    /// # Arguments
    ///
    /// * `padded` - Padded sequences [batch_size, max_len, features]
    /// * `lengths` - Original sequence lengths
    ///
    /// # Returns
    ///
    /// Variable-length sequences without padding
    pub fn unpad_sequences(padded: &Array3<f32>, lengths: &[usize]) -> Vec<Array2<f32>> {
        let (batch_size, _, feature_dim) = padded.dim();
        assert_eq!(batch_size, lengths.len());

        lengths
            .iter()
            .enumerate()
            .map(|(batch_idx, &seq_len)| {
                let mut seq = Array2::zeros((seq_len, feature_dim));
                for t in 0..seq_len {
                    for f in 0..feature_dim {
                        seq[[t, f]] = padded[[batch_idx, t, f]];
                    }
                }
                seq
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba::{Mamba, MambaConfig};

    #[test]
    fn test_batched_model_creation() {
        let config = MambaConfig::default().hidden_dim(64).num_layers(2);
        let model = Mamba::new(config).expect("Failed to create Mamba model");
        let batched = BatchedModel::new(model, 4);
        assert!(batched.is_ok());

        let batched = batched.expect("Failed to create batched model");
        assert_eq!(batched.batch_size(), 4);
    }

    #[test]
    fn test_batch_prediction() {
        let config = MambaConfig::default().hidden_dim(64).num_layers(2);
        let model = Mamba::new(config).expect("Failed to create Mamba model");
        let mut batched = BatchedModel::new(model, 4).expect("Failed to create batched model");

        let inputs = Array2::zeros((4, 1));
        let outputs = batched.predict_batch(&inputs);
        assert!(outputs.is_ok());

        let outputs = outputs.expect("Failed to predict batch");
        assert_eq!(outputs.dim(), (4, 64));
    }

    #[test]
    fn test_sequence_batch_prediction() {
        let config = MambaConfig::default().hidden_dim(64).num_layers(2);
        let model = Mamba::new(config).expect("Failed to create Mamba model");
        let mut batched = BatchedModel::new(model, 2).expect("Failed to create batched model");

        let inputs = Array3::zeros((2, 10, 1));
        let outputs = batched.predict_sequence_batch(&inputs);
        assert!(outputs.is_ok());

        let outputs = outputs.expect("Failed to predict sequence batch");
        assert_eq!(outputs.dim(), (2, 10, 64));
    }

    #[test]
    fn test_batch_reset() {
        let config = MambaConfig::default().hidden_dim(64).num_layers(2);
        let model = Mamba::new(config).expect("Failed to create Mamba model");
        let mut batched = BatchedModel::new(model, 4).expect("Failed to create batched model");

        // Process some inputs
        let inputs = Array2::zeros((4, 1));
        let _ = batched
            .predict_batch(&inputs)
            .expect("Failed to predict batch");

        // Reset specific items
        batched
            .reset_batch_items(&[0, 2])
            .expect("Failed to reset batch items");
        assert_eq!(batched.sequence_lengths()[0], 0);
        assert_eq!(batched.sequence_lengths()[1], 1);
        assert_eq!(batched.sequence_lengths()[2], 0);
        assert_eq!(batched.sequence_lengths()[3], 1);

        // Reset all
        batched.reset_all();
        assert!(batched.sequence_lengths().iter().all(|&len| len == 0));
    }

    #[test]
    fn test_padding() {
        let seq1 = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Failed to create test array");
        let seq2 = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .expect("Failed to create test array");
        let seq3 = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("Failed to create test array");

        let sequences = vec![seq1, seq2, seq3];
        let (padded, lengths) = padding::pad_sequences(&sequences, 0.0);

        assert_eq!(padded.dim(), (3, 5, 2));
        assert_eq!(lengths, vec![3, 5, 2]);

        // Verify unpadding
        let unpadded = padding::unpad_sequences(&padded, &lengths);
        assert_eq!(unpadded.len(), 3);
        assert_eq!(unpadded[0].dim(), (3, 2));
        assert_eq!(unpadded[1].dim(), (5, 2));
        assert_eq!(unpadded[2].dim(), (2, 2));
    }

    #[test]
    fn test_dynamic_batcher() {
        let config = MambaConfig::default();
        let model = Mamba::new(config).expect("Failed to create Mamba model");
        let mut batcher =
            DynamicBatcher::new(model, 1, 16, 1000).expect("Failed to create dynamic batcher");

        assert_eq!(batcher.current_batch_size(), 1);

        // Simulate fast execution
        batcher.update_batch_size(500);
        assert_eq!(batcher.current_batch_size(), 2);

        // Simulate slow execution
        batcher.update_batch_size(3000);
        assert_eq!(batcher.current_batch_size(), 1);
    }
}
