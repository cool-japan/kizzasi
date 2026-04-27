//! Gradient Checkpointing for Memory-Efficient Training
//!
//! Implements activation checkpointing (also known as gradient checkpointing)
//! to trade compute for memory during training. Instead of storing all
//! intermediate activations for the backward pass, only activations at
//! designated checkpoint boundaries are retained. During backpropagation,
//! discarded activations are recomputed from the nearest checkpoint.
//!
//! This technique, introduced by Chen et al. (2016), can reduce memory usage
//! from O(N) to O(sqrt(N)) for N-layer networks at the cost of one
//! additional forward pass.
//!
//! # Example
//!
//! ```rust,ignore
//! use kizzasi_model::gradient_checkpoint::{ActivationCheckpointer, CheckpointConfig};
//! use scirs2_core::ndarray::Array1;
//!
//! let config = CheckpointConfig {
//!     checkpoint_every_n_layers: 3,
//!     max_checkpoints: 10,
//!     use_mixed_precision: false,
//! };
//!
//! let mut checkpointer = ActivationCheckpointer::new(config);
//!
//! let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
//! let layers = vec![0, 1, 2, 3, 4, 5];
//!
//! let output = checkpointer.checkpointed_forward(
//!     &input,
//!     &layers,
//!     |activation, layer_idx| {
//!         // Your layer forward function
//!         Ok(activation.mapv(|x| x * 1.1 + layer_idx as f32 * 0.01))
//!     },
//! )?;
//! ```

use crate::error::{ModelError, ModelResult};
use scirs2_core::ndarray::Array1;

// ---------------------------------------------------------------------------
// CheckpointConfig
// ---------------------------------------------------------------------------

/// Configuration for activation checkpointing.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Save an activation checkpoint every N layers.
    ///
    /// For example, `3` means layers 0, 3, 6, 9, ... will have their
    /// activations stored.
    pub checkpoint_every_n_layers: usize,

    /// Maximum number of checkpoint slots.
    ///
    /// Once this limit is reached, older checkpoints may be evicted.
    pub max_checkpoints: usize,

    /// Whether to use reduced precision (f16-like truncation) for stored
    /// activations to further save memory.
    ///
    /// When enabled, stored activations are quantised to half-precision
    /// (simulated by rounding to 3 decimal places) and restored on retrieval.
    pub use_mixed_precision: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_every_n_layers: 4,
            max_checkpoints: 64,
            use_mixed_precision: false,
        }
    }
}

// ---------------------------------------------------------------------------
// ActivationCheckpointer
// ---------------------------------------------------------------------------

/// Manages activation checkpoints for a multi-layer forward pass.
///
/// Stores activations at designated layer boundaries so they can be
/// used during backpropagation without keeping every intermediate result
/// in memory.
#[derive(Debug, Clone)]
pub struct ActivationCheckpointer {
    config: CheckpointConfig,
    /// Stored activations indexed by layer. `None` means the activation
    /// for that layer was not checkpointed.
    checkpoints: Vec<Option<Array1<f32>>>,
    /// Tracks total bytes of activations that were *not* stored due to
    /// the checkpointing policy — i.e., the memory that was saved.
    bytes_saved: usize,
    /// Tracks total bytes of activations that *are* stored.
    bytes_stored: usize,
}

impl ActivationCheckpointer {
    /// Create a new checkpointer with the given configuration.
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            checkpoints: Vec::new(),
            bytes_saved: 0,
            bytes_stored: 0,
        }
    }

    /// Save an activation at the given layer index.
    ///
    /// If `use_mixed_precision` is enabled, the activation is quantised
    /// before storage.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_checkpoints` would be exceeded and the
    /// layer is not a checkpoint boundary.
    pub fn save(&mut self, layer_idx: usize, activation: Array1<f32>) -> ModelResult<()> {
        // Ensure the checkpoints vector is large enough.
        if layer_idx >= self.checkpoints.len() {
            self.checkpoints.resize(layer_idx + 1, None);
        }

        // Check checkpoint capacity.
        let current_count = self.num_checkpoints();
        if current_count >= self.config.max_checkpoints && self.checkpoints[layer_idx].is_none() {
            return Err(ModelError::invalid_config(format!(
                "Maximum checkpoint count ({}) exceeded when saving layer {}",
                self.config.max_checkpoints, layer_idx
            )));
        }

        let byte_size = activation.len() * std::mem::size_of::<f32>();

        let stored = if self.config.use_mixed_precision {
            // Simulate half-precision by rounding to 3 decimal places.
            // This halves effective precision while keeping f32 storage format.
            activation.mapv(|x| (x * 1000.0).round() / 1000.0)
        } else {
            activation
        };

        self.bytes_stored += byte_size;
        self.checkpoints[layer_idx] = Some(stored);

        Ok(())
    }

    /// Retrieve the checkpointed activation at the given layer.
    ///
    /// # Errors
    ///
    /// Returns an error if no checkpoint exists for `layer_idx`.
    pub fn get(&self, layer_idx: usize) -> ModelResult<&Array1<f32>> {
        if layer_idx >= self.checkpoints.len() {
            return Err(ModelError::IndexOutOfBounds {
                index: layer_idx,
                limit: self.checkpoints.len(),
                context: "ActivationCheckpointer::get".to_string(),
            });
        }

        self.checkpoints[layer_idx].as_ref().ok_or_else(|| {
            ModelError::not_initialized(format!("No checkpoint stored for layer {}", layer_idx))
        })
    }

    /// Clear all stored checkpoints and reset memory accounting.
    pub fn clear(&mut self) {
        self.checkpoints.clear();
        self.bytes_saved = 0;
        self.bytes_stored = 0;
    }

    /// Estimated bytes of memory saved by not storing non-checkpointed
    /// activations.
    ///
    /// This value is updated during `checkpointed_forward` calls.
    pub fn memory_saved_bytes(&self) -> usize {
        self.bytes_saved
    }

    /// Bytes currently stored in checkpoints.
    pub fn memory_stored_bytes(&self) -> usize {
        self.bytes_stored
    }

    /// Number of non-`None` checkpoints currently held.
    pub fn num_checkpoints(&self) -> usize {
        self.checkpoints.iter().filter(|c| c.is_some()).count()
    }

    /// Whether a given layer index is a checkpoint boundary according to
    /// the current configuration.
    pub fn is_checkpoint_layer(&self, layer_idx: usize) -> bool {
        if self.config.checkpoint_every_n_layers == 0 {
            return false;
        }
        layer_idx.is_multiple_of(self.config.checkpoint_every_n_layers)
    }

    /// Run a checkpointed forward pass through the given layers.
    ///
    /// The `forward_fn` is called sequentially for each layer in `layers`,
    /// receiving the current activation and the layer index. Activations
    /// at checkpoint boundaries are saved; others are discarded (their
    /// memory cost is recorded in `bytes_saved`).
    ///
    /// # Parameters
    ///
    /// - `input`: the initial activation fed into the first layer.
    /// - `layers`: ordered list of layer indices to process.
    /// - `forward_fn`: `Fn(&Array1<f32>, usize) -> ModelResult<Array1<f32>>`;
    ///   applies one layer's computation.
    ///
    /// # Returns
    ///
    /// The activation after all layers have been applied.
    pub fn checkpointed_forward<F>(
        &mut self,
        input: &Array1<f32>,
        layers: &[usize],
        forward_fn: F,
    ) -> ModelResult<Array1<f32>>
    where
        F: Fn(&Array1<f32>, usize) -> ModelResult<Array1<f32>>,
    {
        let mut current = input.clone();

        for &layer_idx in layers {
            current = forward_fn(&current, layer_idx)?;

            let byte_size = current.len() * std::mem::size_of::<f32>();

            if self.is_checkpoint_layer(layer_idx) {
                // Save checkpoint (respects max_checkpoints internally).
                if self.num_checkpoints() < self.config.max_checkpoints {
                    self.save(layer_idx, current.clone())?;
                } else {
                    // Cannot save more — count as saved memory.
                    self.bytes_saved += byte_size;
                }
            } else {
                // Not a checkpoint layer — activation is discarded.
                self.bytes_saved += byte_size;
            }
        }

        Ok(current)
    }

    /// Recompute activations from the nearest checkpoint up to `target_layer`.
    ///
    /// This is used during the backward pass: find the closest checkpoint
    /// before `target_layer`, then replay the forward function from there.
    ///
    /// # Parameters
    ///
    /// - `target_layer`: the layer whose activation is needed.
    /// - `layers`: the full ordered list of layer indices.
    /// - `forward_fn`: the same forward function used during the forward pass.
    ///
    /// # Returns
    ///
    /// The recomputed activation at `target_layer`.
    pub fn recompute_from_checkpoint<F>(
        &self,
        target_layer: usize,
        layers: &[usize],
        forward_fn: F,
    ) -> ModelResult<Array1<f32>>
    where
        F: Fn(&Array1<f32>, usize) -> ModelResult<Array1<f32>>,
    {
        // Find the nearest checkpoint at or before target_layer.
        let mut nearest_checkpoint_layer = None;
        let mut nearest_activation = None;

        for &l in layers.iter().rev() {
            if l > target_layer {
                continue;
            }
            if l < self.checkpoints.len() {
                if let Some(ref act) = self.checkpoints[l] {
                    nearest_checkpoint_layer = Some(l);
                    nearest_activation = Some(act.clone());
                    break;
                }
            }
        }

        let (start_layer, mut current) = match (nearest_checkpoint_layer, nearest_activation) {
            (Some(l), Some(act)) => (l, act),
            _ => {
                return Err(ModelError::not_initialized(format!(
                    "No checkpoint found before layer {} for recomputation",
                    target_layer
                )));
            }
        };

        // Replay forward from the checkpoint layer to the target.
        let mut started = false;
        for &l in layers {
            if l == start_layer {
                started = true;
                continue; // Skip the checkpoint layer itself — we already have its activation.
            }
            if !started {
                continue;
            }
            current = forward_fn(&current, l)?;
            if l == target_layer {
                break;
            }
        }

        Ok(current)
    }

    /// Return the configuration.
    pub fn config(&self) -> &CheckpointConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    /// Simple forward function: multiplies each element by 1.1 and adds
    /// a layer-dependent offset.
    fn simple_forward(activation: &Array1<f32>, layer_idx: usize) -> ModelResult<Array1<f32>> {
        Ok(activation.mapv(|x| x * 1.1 + layer_idx as f32 * 0.01))
    }

    // 6. Save and get
    #[test]
    fn test_gradient_checkpoint_save_get() {
        let config = CheckpointConfig {
            checkpoint_every_n_layers: 2,
            max_checkpoints: 10,
            use_mixed_precision: false,
        };
        let mut cp = ActivationCheckpointer::new(config);

        let activation = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        cp.save(2, activation.clone()).expect("save should succeed");

        let retrieved = cp.get(2).expect("get should succeed");
        assert_eq!(retrieved.len(), 4);
        assert!((retrieved[0] - 1.0).abs() < 1e-6);
        assert!((retrieved[3] - 4.0).abs() < 1e-6);

        // Getting a non-existent layer should fail.
        assert!(cp.get(5).is_err());
    }

    // 7. Memory accounting
    #[test]
    fn test_gradient_checkpoint_memory_accounting() {
        let config = CheckpointConfig {
            checkpoint_every_n_layers: 3,
            max_checkpoints: 10,
            use_mixed_precision: false,
        };
        let mut cp = ActivationCheckpointer::new(config);

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let layers: Vec<usize> = (0..6).collect();

        let _output = cp
            .checkpointed_forward(&input, &layers, simple_forward)
            .expect("forward should succeed");

        // Checkpoints at layers 0, 3 (divisible by 3)
        // Non-checkpointed: layers 1, 2, 4, 5
        assert!(
            cp.memory_saved_bytes() > 0,
            "should have saved some memory, got 0"
        );
        assert!(
            cp.memory_stored_bytes() > 0,
            "should have stored some activations"
        );
        assert_eq!(
            cp.num_checkpoints(),
            2,
            "should have 2 checkpoints (layers 0, 3)"
        );
    }

    // 8. Clear resets everything
    #[test]
    fn test_gradient_checkpoint_clear() {
        let config = CheckpointConfig {
            checkpoint_every_n_layers: 2,
            max_checkpoints: 10,
            use_mixed_precision: false,
        };
        let mut cp = ActivationCheckpointer::new(config);

        cp.save(0, Array1::from_vec(vec![1.0, 2.0]))
            .expect("save should succeed");
        cp.save(2, Array1::from_vec(vec![3.0, 4.0]))
            .expect("save should succeed");

        assert_eq!(cp.num_checkpoints(), 2);

        cp.clear();

        assert_eq!(cp.num_checkpoints(), 0);
        assert_eq!(cp.memory_saved_bytes(), 0);
        assert_eq!(cp.memory_stored_bytes(), 0);
        assert!(cp.get(0).is_err());
    }

    // 9. Checkpointed forward produces same output as direct sequential forward
    #[test]
    fn test_gradient_checkpoint_forward() {
        let config = CheckpointConfig {
            checkpoint_every_n_layers: 2,
            max_checkpoints: 20,
            use_mixed_precision: false,
        };
        let mut cp = ActivationCheckpointer::new(config);

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let layers: Vec<usize> = (0..8).collect();

        // Checkpointed forward
        let checkpointed_output = cp
            .checkpointed_forward(&input, &layers, simple_forward)
            .expect("checkpointed forward should succeed");

        // Direct sequential forward (no checkpointing)
        let mut direct = input.clone();
        for &l in &layers {
            direct = simple_forward(&direct, l).expect("forward should succeed");
        }

        // Both should produce the same result
        assert_eq!(checkpointed_output.len(), direct.len());
        for (a, b) in checkpointed_output.iter().zip(direct.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "mismatch: checkpointed={}, direct={}",
                a,
                b
            );
        }

        // Verify some checkpoints were actually saved
        assert!(cp.num_checkpoints() > 0, "should have saved checkpoints");
    }
}
