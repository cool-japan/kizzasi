//! DataLoader for time-series training
//!
//! Provides efficient data loading, batching, and preprocessing for time-series
//! signal prediction tasks.
//!
//! # Features
//!
//! - **Windowing**: Sliding window extraction from continuous signals
//! - **Batching**: Efficient mini-batch creation with shuffling
//! - **Prefetching**: Async data loading for GPU transfer
//! - **Augmentation**: Time-series specific augmentations
//! - **Multi-GPU**: Distributed data loading support
//!
//! # Examples
//!
//! ```rust
//! use kizzasi_core::dataloader::{TimeSeriesDataLoader, DataLoaderConfig};
//! use scirs2_core::ndarray::Array2;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let data = Array2::<f32>::zeros((1000, 3));  // 1000 timesteps, 3 features
//! let config = DataLoaderConfig::default()
//!     .with_window_size(64)
//!     .with_batch_size(32)
//!     .with_shuffle(true);
//!
//! let mut loader = TimeSeriesDataLoader::new(data, config)?;
//!
//! for batch in loader.iter_batches() {
//!     let (inputs, targets) = batch?;
//!     // Train on batch
//! }
//! # Ok(())
//! # }
//! ```

use crate::error::{CoreError, CoreResult};
use candle_core::{Device, Tensor};
use scirs2_core::ndarray::{s, Array2};
use serde::{Deserialize, Serialize};

/// Configuration for time-series data loader
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLoaderConfig {
    /// Window size for sliding window extraction
    pub window_size: usize,
    /// Prediction horizon (number of steps ahead to predict)
    pub horizon: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Whether to shuffle data
    pub shuffle: bool,
    /// Overlap between consecutive windows (0.0 = no overlap, 0.5 = 50% overlap)
    pub overlap: f32,
    /// Whether to drop last incomplete batch
    pub drop_last: bool,
    /// Number of workers for parallel loading (future)
    pub num_workers: usize,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            window_size: 64,
            horizon: 1,
            batch_size: 32,
            shuffle: true,
            overlap: 0.0,
            drop_last: false,
            num_workers: 1,
        }
    }
}

impl DataLoaderConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    pub fn with_horizon(mut self, horizon: usize) -> Self {
        self.horizon = horizon;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn with_overlap(mut self, overlap: f32) -> Self {
        self.overlap = overlap.clamp(0.0, 1.0);
        self
    }

    pub fn with_drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }
}

/// Time-series data loader
pub struct TimeSeriesDataLoader {
    data: Array2<f32>,
    config: DataLoaderConfig,
    indices: Vec<usize>,
    current_epoch: usize,
}

impl TimeSeriesDataLoader {
    /// Create a new data loader
    ///
    /// # Arguments
    /// * `data` - Time-series data of shape [timesteps, features]
    /// * `config` - DataLoader configuration
    pub fn new(data: Array2<f32>, config: DataLoaderConfig) -> CoreResult<Self> {
        if data.nrows() < config.window_size + config.horizon {
            return Err(CoreError::InvalidConfig(format!(
                "Data length {} is too short for window_size {} + horizon {}",
                data.nrows(),
                config.window_size,
                config.horizon
            )));
        }

        // Calculate stride based on overlap
        let stride = ((config.window_size as f32) * (1.0 - config.overlap)).max(1.0) as usize;

        // Generate window start indices
        let max_start = data.nrows() - config.window_size - config.horizon + 1;
        let indices: Vec<usize> = (0..max_start).step_by(stride).collect();

        Ok(Self {
            data,
            config,
            indices,
            current_epoch: 0,
        })
    }

    /// Split the series chronologically into `(train, validation)` loaders.
    ///
    /// The **tail** `val_ratio` fraction of the raw time series becomes the
    /// validation half. The split is positional, never random: a shuffled
    /// split of sliding windows would let the model validate on windows that
    /// overlap — and therefore leak — its own training data.
    ///
    /// Windows never cross the cut point, so the `window_size + horizon`
    /// timesteps straddling the boundary are dropped rather than shared.
    ///
    /// # Arguments
    /// * `val_ratio` - fraction of timesteps reserved for validation, in `(0, 1)`
    /// * `val_batch_size` - batch size for the validation loader; shuffling is
    ///   always disabled on the validation half
    ///
    /// # Errors
    /// Returns [`CoreError::InvalidConfig`] when `val_ratio` is outside
    /// `(0, 1)`, when `val_batch_size` is zero, or when the series is too short
    /// for both halves to yield at least one window.
    pub fn split_chronological(
        &self,
        val_ratio: f32,
        val_batch_size: usize,
    ) -> CoreResult<(Self, Self)> {
        if !val_ratio.is_finite() || val_ratio <= 0.0 || val_ratio >= 1.0 {
            return Err(CoreError::InvalidConfig(format!(
                "validation split must lie in the open interval (0, 1), got {}",
                val_ratio
            )));
        }
        if val_batch_size == 0 {
            return Err(CoreError::InvalidConfig(
                "validation batch size must be greater than zero".to_string(),
            ));
        }

        let total_rows = self.data.nrows();
        let min_rows = self.config.window_size + self.config.horizon;
        if total_rows < 2 * min_rows {
            return Err(CoreError::InvalidConfig(format!(
                "series of {} timesteps cannot be split: each half needs at least {} \
                 (window_size {} + horizon {})",
                total_rows, min_rows, self.config.window_size, self.config.horizon
            )));
        }

        // Clamp so both halves keep at least one full window; the bounds are
        // valid because `total_rows >= 2 * min_rows` was checked above.
        let requested = ((total_rows as f32) * val_ratio).round() as usize;
        let val_rows = requested.clamp(min_rows, total_rows - min_rows);
        let cut = total_rows - val_rows;

        let train_data = self.data.slice(s![..cut, ..]).to_owned();
        let val_data = self.data.slice(s![cut.., ..]).to_owned();

        let train = Self::new(train_data, self.config.clone())?;
        let val_config = DataLoaderConfig {
            batch_size: val_batch_size,
            shuffle: false,
            ..self.config.clone()
        };
        let val = Self::new(val_data, val_config)?;

        Ok((train, val))
    }

    /// Get number of batches per epoch
    pub fn num_batches(&self) -> usize {
        let num_samples = self.indices.len();
        // A zero batch size would divide by zero; treat it as "one sample per
        // batch" so a misconfigured loader degrades instead of panicking.
        let batch_size = self.config.batch_size.max(1);
        if self.config.drop_last {
            num_samples / batch_size
        } else {
            num_samples.div_ceil(batch_size)
        }
    }

    /// Get total number of samples
    pub fn num_samples(&self) -> usize {
        self.indices.len()
    }

    /// Shuffle indices for new epoch
    pub fn shuffle(&mut self) {
        if self.config.shuffle {
            use scirs2_core::convenience::uniform;
            // Fisher-Yates shuffle
            for i in (1..self.indices.len()).rev() {
                let j = (uniform() * (i + 1) as f64) as usize;
                self.indices.swap(i, j);
            }
        }
    }

    /// Extract a single window
    fn extract_window(&self, start_idx: usize) -> CoreResult<(Array2<f32>, Array2<f32>)> {
        let end_input = start_idx + self.config.window_size;
        let end_target = end_input + self.config.horizon;

        if end_target > self.data.nrows() {
            return Err(CoreError::Generic(format!(
                "Window exceeds data bounds: {} > {}",
                end_target,
                self.data.nrows()
            )));
        }

        let input = self.data.slice(s![start_idx..end_input, ..]).to_owned();

        let target = self.data.slice(s![end_input..end_target, ..]).to_owned();

        Ok((input, target))
    }

    /// Create a batch of windows
    fn create_batch(&self, batch_indices: &[usize]) -> CoreResult<(Array2<f32>, Array2<f32>)> {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for &idx in batch_indices {
            let start = self.indices[idx];
            let (input, target) = self.extract_window(start)?;
            inputs.push(input);
            targets.push(target);
        }

        // Stack into batch: [batch, time, features]
        let batch_size = inputs.len();
        let window_size = self.config.window_size;
        let horizon = self.config.horizon;
        let n_features = self.data.ncols();

        let mut batch_input = Array2::zeros((batch_size * window_size, n_features));
        let mut batch_target = Array2::zeros((batch_size * horizon, n_features));

        for (i, (inp, tgt)) in inputs.iter().zip(targets.iter()).enumerate() {
            let input_start = i * window_size;
            let input_end = input_start + window_size;
            batch_input
                .slice_mut(s![input_start..input_end, ..])
                .assign(inp);

            let target_start = i * horizon;
            let target_end = target_start + horizon;
            batch_target
                .slice_mut(s![target_start..target_end, ..])
                .assign(tgt);
        }

        Ok((batch_input, batch_target))
    }

    /// Iterate over batches
    pub fn iter_batches(&mut self) -> BatchIterator<'_> {
        if self.current_epoch > 0 {
            self.shuffle();
        }
        self.current_epoch += 1;

        BatchIterator {
            loader: self,
            current_batch: 0,
        }
    }

    /// Convert batch to candle tensors
    pub fn to_tensors(
        &self,
        inputs: &Array2<f32>,
        targets: &Array2<f32>,
        device: &Device,
    ) -> CoreResult<(Tensor, Tensor)> {
        let batch_size = inputs.nrows() / self.config.window_size;
        let window_size = self.config.window_size;
        let horizon = self.config.horizon;
        let n_features = inputs.ncols();

        // Flatten to Vec<f32>
        let input_vec: Vec<f32> = inputs.iter().copied().collect();
        let target_vec: Vec<f32> = targets.iter().copied().collect();

        // Create tensors with shape [batch, seq, features]
        let input_tensor =
            Tensor::from_vec(input_vec, &[batch_size, window_size, n_features], device)
                .map_err(|e| CoreError::Generic(format!("Failed to create input tensor: {}", e)))?;

        let target_tensor =
            Tensor::from_vec(target_vec, &[batch_size, horizon, n_features], device).map_err(
                |e| CoreError::Generic(format!("Failed to create target tensor: {}", e)),
            )?;

        Ok((input_tensor, target_tensor))
    }

    /// Get configuration
    pub fn config(&self) -> &DataLoaderConfig {
        &self.config
    }
}

/// Iterator over batches
pub struct BatchIterator<'a> {
    loader: &'a TimeSeriesDataLoader,
    current_batch: usize,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = CoreResult<(Array2<f32>, Array2<f32>)>;

    fn next(&mut self) -> Option<Self::Item> {
        let num_batches = self.loader.num_batches();
        if self.current_batch >= num_batches {
            return None;
        }

        // Mirror `num_batches`: a zero batch size degrades to one sample per
        // batch instead of yielding empty batches forever.
        let batch_size = self.loader.config.batch_size.max(1);
        let start_idx = self.current_batch * batch_size;
        let end_idx = (start_idx + batch_size).min(self.loader.indices.len());

        // Check if we should drop last incomplete batch
        if self.loader.config.drop_last && end_idx.saturating_sub(start_idx) < batch_size {
            return None;
        }

        let batch_indices: Vec<usize> = (start_idx..end_idx).collect();
        self.current_batch += 1;

        Some(self.loader.create_batch(&batch_indices))
    }
}

/// Data augmentation for time-series
pub struct TimeSeriesAugmentation;

impl TimeSeriesAugmentation {
    /// Add Gaussian noise
    pub fn add_noise(data: &Array2<f32>, std: f32) -> Array2<f32> {
        use scirs2_core::convenience::uniform;
        let noise = Array2::from_shape_fn(data.dim(), |_| {
            // Box-Muller transform for Gaussian
            let u1 = uniform();
            let u2 = uniform();
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            (z0 * std as f64) as f32
        });
        data + &noise
    }

    /// Scale by random factor
    pub fn scale(data: &Array2<f32>, min_scale: f32, max_scale: f32) -> Array2<f32> {
        use scirs2_core::convenience::uniform;
        let scale = uniform() * (max_scale - min_scale) as f64 + min_scale as f64;
        data * (scale as f32)
    }

    /// Time shift (circular shift along time axis)
    pub fn time_shift(data: &Array2<f32>, max_shift: usize) -> Array2<f32> {
        use scirs2_core::convenience::uniform;
        let shift = (uniform() * max_shift as f64) as usize;

        let mut shifted = data.clone();
        if shift > 0 {
            let n = data.nrows();
            for i in 0..n {
                let src = (i + shift) % n;
                shifted.row_mut(i).assign(&data.row(src));
            }
        }
        shifted
    }

    /// Apply random masking (set random timesteps to zero)
    pub fn mask(data: &Array2<f32>, mask_prob: f32) -> Array2<f32> {
        use scirs2_core::convenience::uniform;
        let mut masked = data.clone();
        for i in 0..masked.nrows() {
            if uniform() < mask_prob as f64 {
                masked.row_mut(i).fill(0.0);
            }
        }
        masked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataloader_creation() {
        let data = Array2::<f32>::zeros((1000, 3));
        let config = DataLoaderConfig::default()
            .with_window_size(64)
            .with_batch_size(32);

        let loader = TimeSeriesDataLoader::new(data, config);
        assert!(loader.is_ok());
    }

    #[test]
    fn test_dataloader_insufficient_data() {
        let data = Array2::<f32>::zeros((50, 3)); // Too short
        let config = DataLoaderConfig::default()
            .with_window_size(64)
            .with_horizon(1);

        let loader = TimeSeriesDataLoader::new(data, config);
        assert!(loader.is_err());
    }

    #[test]
    fn test_num_batches() {
        let data = Array2::<f32>::zeros((1000, 3));
        let config = DataLoaderConfig::default()
            .with_window_size(64)
            .with_batch_size(32)
            .with_overlap(0.0);

        let loader = TimeSeriesDataLoader::new(data, config).unwrap();
        assert!(loader.num_batches() > 0);
    }

    #[test]
    fn test_batch_iteration() {
        let data = Array2::<f32>::from_shape_fn((200, 3), |(i, j)| (i + j) as f32);
        let config = DataLoaderConfig::default()
            .with_window_size(10)
            .with_batch_size(4)
            .with_horizon(1)
            .with_shuffle(false);

        let mut loader = TimeSeriesDataLoader::new(data, config).unwrap();

        let mut batch_count = 0;
        for batch in loader.iter_batches() {
            let (inputs, targets) = batch.unwrap();
            assert_eq!(inputs.ncols(), 3);
            assert_eq!(targets.ncols(), 3);
            batch_count += 1;
        }

        assert!(batch_count > 0);
        assert_eq!(batch_count, loader.num_batches());
    }

    #[test]
    fn test_tensor_conversion() {
        let data = Array2::<f32>::from_shape_fn((200, 3), |(i, j)| (i + j) as f32);
        let config = DataLoaderConfig::default()
            .with_window_size(10)
            .with_batch_size(4)
            .with_horizon(1);

        let mut loader = TimeSeriesDataLoader::new(data, config).unwrap();

        // Test just one batch
        let batch = loader.iter_batches().next().unwrap();
        let (inputs, targets) = batch.unwrap();
        let device = Device::Cpu;

        let (input_tensor, target_tensor) = loader.to_tensors(&inputs, &targets, &device).unwrap();

        assert_eq!(input_tensor.dims().len(), 3); // [batch, seq, features]
        assert_eq!(target_tensor.dims().len(), 3);
        assert_eq!(input_tensor.dims()[2], 3); // 3 features
    }

    #[test]
    fn test_overlap() {
        let data = Array2::<f32>::zeros((200, 3));
        let config_no_overlap = DataLoaderConfig::default()
            .with_window_size(10)
            .with_overlap(0.0);

        let config_overlap = DataLoaderConfig::default()
            .with_window_size(10)
            .with_overlap(0.5);

        let loader_no_overlap = TimeSeriesDataLoader::new(data.clone(), config_no_overlap).unwrap();
        let loader_overlap = TimeSeriesDataLoader::new(data, config_overlap).unwrap();

        // With overlap, we should have more samples
        assert!(loader_overlap.num_samples() > loader_no_overlap.num_samples());
    }

    #[test]
    fn test_augmentation_noise() {
        let data = Array2::<f32>::zeros((100, 3));
        let augmented = TimeSeriesAugmentation::add_noise(&data, 0.1);

        assert_eq!(augmented.dim(), data.dim());
        // With noise, not all values should be exactly zero
        assert!(augmented.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_augmentation_scale() {
        let data = Array2::<f32>::ones((100, 3));
        let augmented = TimeSeriesAugmentation::scale(&data, 0.5, 1.5);

        assert_eq!(augmented.dim(), data.dim());
        // Values should be scaled
        let mean = augmented.mean().unwrap();
        assert!((0.5..=1.5).contains(&mean));
    }

    #[test]
    fn test_drop_last() {
        let data = Array2::<f32>::zeros((100, 3));
        let config_drop = DataLoaderConfig::default()
            .with_window_size(10)
            .with_batch_size(7)
            .with_drop_last(true);

        let config_no_drop = DataLoaderConfig::default()
            .with_window_size(10)
            .with_batch_size(7)
            .with_drop_last(false);

        let loader_drop = TimeSeriesDataLoader::new(data.clone(), config_drop).unwrap();
        let loader_no_drop = TimeSeriesDataLoader::new(data, config_no_drop).unwrap();

        // Without drop_last, we might have more batches
        assert!(loader_no_drop.num_batches() >= loader_drop.num_batches());
    }

    #[test]
    fn test_split_chronological_takes_the_tail_for_validation() {
        // Row `t` is filled with the value `t` so the split point is visible in
        // the data itself: the validation half must contain only later
        // timesteps than the training half.
        let n = 200usize;
        let data = Array2::from_shape_fn((n, 1), |(t, _)| t as f32);

        let config = DataLoaderConfig::default()
            .with_window_size(8)
            .with_horizon(4)
            .with_batch_size(4)
            .with_shuffle(true);
        let loader = TimeSeriesDataLoader::new(data, config).unwrap();

        let (train, val) = loader.split_chronological(0.25, 2).unwrap();

        assert_eq!(
            val.config().batch_size,
            2,
            "validation batch size must come from the argument"
        );
        assert!(!val.config().shuffle, "validation must not be shuffled");
        assert_eq!(
            train.config().batch_size,
            4,
            "train keeps its own batch size"
        );
        assert!(train.num_batches() > 0 && val.num_batches() > 0);

        let train_max = train.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let val_min = val.data.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            val_min > train_max,
            "validation window must start strictly after the training window \
             (train_max={train_max}, val_min={val_min})"
        );
        assert_eq!(
            train.data.nrows() + val.data.nrows(),
            n,
            "the split must partition the series"
        );
    }

    #[test]
    fn test_split_chronological_rejects_invalid_input() {
        let data = Array2::<f32>::zeros((200, 1));
        let config = DataLoaderConfig::default()
            .with_window_size(8)
            .with_horizon(4)
            .with_batch_size(4);
        let loader = TimeSeriesDataLoader::new(data, config).unwrap();

        assert!(loader.split_chronological(0.0, 4).is_err());
        assert!(loader.split_chronological(1.0, 4).is_err());
        assert!(loader.split_chronological(-0.5, 4).is_err());
        assert!(loader.split_chronological(f32::NAN, 4).is_err());
        assert!(loader.split_chronological(0.2, 0).is_err());

        // A series too short for two windows must report an error, not a
        // silently degenerate loader.
        let short = Array2::<f32>::zeros((20, 1));
        let short_config = DataLoaderConfig::default()
            .with_window_size(8)
            .with_horizon(4)
            .with_batch_size(4);
        let short_loader = TimeSeriesDataLoader::new(short, short_config).unwrap();
        assert!(short_loader.split_chronological(0.2, 4).is_err());
    }

    #[test]
    fn test_split_chronological_clamps_extreme_ratios() {
        // A ratio that would leave one half without a full window is clamped so
        // both halves stay usable.
        let data = Array2::<f32>::zeros((60, 1));
        let config = DataLoaderConfig::default()
            .with_window_size(8)
            .with_horizon(4)
            .with_batch_size(2);
        let loader = TimeSeriesDataLoader::new(data, config).unwrap();

        let (train, val) = loader.split_chronological(0.99, 2).unwrap();
        assert!(train.data.nrows() >= 12);
        assert!(val.data.nrows() >= 12);
        assert!(train.num_samples() > 0 && val.num_samples() > 0);
    }
}
