//! Causal convolution implementations for SSM architectures
//!
//! Causal convolutions are essential for autoregressive models as they
//! ensure the output at time t only depends on inputs at times <= t.

use crate::error::{CoreError, CoreResult};
use scirs2_core::ndarray::Array1;

/// 1D Causal Convolution Layer
///
/// Implements a causal (left-padded) convolution that preserves causality
/// for autoregressive inference. Used as the input projection in Mamba.
#[derive(Debug, Clone)]
pub struct CausalConv1d {
    /// Convolution kernel weights [out_channels, in_channels, kernel_size]
    weights: Vec<Vec<Vec<f32>>>,
    /// Bias per output channel
    bias: Vec<f32>,
    /// Kernel size (also determines padding)
    kernel_size: usize,
    /// Input channels
    in_channels: usize,
    /// Output channels
    out_channels: usize,
    /// Ring buffer for causal history
    history: Vec<Vec<f32>>,
}

impl CausalConv1d {
    /// Create a new causal convolution
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        // Initialize weights with small random values using Kaiming init
        let scale = (2.0 / (in_channels * kernel_size) as f32).sqrt();
        let mut weights = Vec::with_capacity(out_channels);

        for _ in 0..out_channels {
            let mut out_ch = Vec::with_capacity(in_channels);
            for _ in 0..in_channels {
                let kernel: Vec<f32> = (0..kernel_size)
                    .map(|i| {
                        // Simple deterministic initialization
                        (i as f32 * 0.1).sin() * scale
                    })
                    .collect();
                out_ch.push(kernel);
            }
            weights.push(out_ch);
        }

        let bias = vec![0.0; out_channels];

        // Initialize history buffer with zeros (kernel_size - 1 frames needed for causality)
        let history: Vec<Vec<f32>> = (0..(kernel_size - 1))
            .map(|_| vec![0.0; in_channels])
            .collect();

        Self {
            weights,
            bias,
            kernel_size,
            in_channels,
            out_channels,
            history,
        }
    }

    /// Set weights from external source.
    ///
    /// # Panics
    /// Panics if `weights` does not match `[out_channels, in_channels,
    /// kernel_size]`. `set_weights` is the crate's established (and
    /// externally-used) infallible API; use [`Self::set_weights_checked`]
    /// for a non-panicking variant that returns
    /// [`CoreError::DimensionMismatch`] instead.
    pub fn set_weights(&mut self, weights: Vec<Vec<Vec<f32>>>) {
        self.set_weights_checked(weights)
            .unwrap_or_else(|e| panic!("CausalConv1d::set_weights: {e}"));
    }

    /// Set weights from external source, returning
    /// [`CoreError::DimensionMismatch`] on a shape mismatch instead of
    /// panicking.
    pub fn set_weights_checked(&mut self, weights: Vec<Vec<Vec<f32>>>) -> CoreResult<()> {
        if weights.len() != self.out_channels {
            return Err(CoreError::DimensionMismatch {
                expected: self.out_channels,
                got: weights.len(),
            });
        }
        for oc in &weights {
            if oc.len() != self.in_channels {
                return Err(CoreError::DimensionMismatch {
                    expected: self.in_channels,
                    got: oc.len(),
                });
            }
            for ic in oc {
                if ic.len() != self.kernel_size {
                    return Err(CoreError::DimensionMismatch {
                        expected: self.kernel_size,
                        got: ic.len(),
                    });
                }
            }
        }
        self.weights = weights;
        Ok(())
    }

    /// Set bias from external source.
    ///
    /// # Panics
    /// Panics if `bias.len() != out_channels`. Use [`Self::set_bias_checked`]
    /// for a non-panicking variant.
    pub fn set_bias(&mut self, bias: Vec<f32>) {
        self.set_bias_checked(bias)
            .unwrap_or_else(|e| panic!("CausalConv1d::set_bias: {e}"));
    }

    /// Set bias from external source, returning
    /// [`CoreError::DimensionMismatch`] on a length mismatch instead of
    /// panicking.
    pub fn set_bias_checked(&mut self, bias: Vec<f32>) -> CoreResult<()> {
        if bias.len() != self.out_channels {
            return Err(CoreError::DimensionMismatch {
                expected: self.out_channels,
                got: bias.len(),
            });
        }
        self.bias = bias;
        Ok(())
    }

    /// Forward pass for a single time step (streaming/causal).
    ///
    /// Takes input of shape `[in_channels]` and returns output of shape `[out_channels]`.
    ///
    /// # Panics
    /// Panics if `input.len() != in_channels`. Use
    /// [`Self::forward_step_checked`] for a non-panicking variant.
    pub fn forward_step(&mut self, input: &[f32]) -> Vec<f32> {
        self.forward_step_checked(input)
            .unwrap_or_else(|e| panic!("CausalConv1d::forward_step: {e}"))
    }

    /// Forward pass for a single time step, returning
    /// [`CoreError::DimensionMismatch`] if `input.len() != in_channels`
    /// instead of panicking.
    pub fn forward_step_checked(&mut self, input: &[f32]) -> CoreResult<Vec<f32>> {
        if input.len() != self.in_channels {
            return Err(CoreError::DimensionMismatch {
                expected: self.in_channels,
                got: input.len(),
            });
        }

        // Add current input to history
        self.history.push(input.to_vec());

        // Keep only kernel_size frames
        while self.history.len() > self.kernel_size {
            self.history.remove(0);
        }

        // Compute convolution output
        let mut output = self.bias.clone();

        for (oc, out_weights) in self.weights.iter().enumerate() {
            for (ic, in_weights) in out_weights.iter().enumerate() {
                for (k, &weight) in in_weights.iter().enumerate() {
                    // For causal conv, we use history[0..kernel_size]
                    // where history[kernel_size-1] is the current input
                    if k < self.history.len() {
                        let hist_idx = self.history.len() - 1 - k;
                        output[oc] += weight * self.history[hist_idx][ic];
                    }
                }
            }
        }

        Ok(output)
    }

    /// Forward pass for a batch of time steps
    ///
    /// Input shape: [time, in_channels]
    /// Output shape: [time, out_channels]
    pub fn forward_batch(&mut self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        input.iter().map(|x| self.forward_step(x)).collect()
    }

    /// Reset the history buffer
    pub fn reset(&mut self) {
        for h in &mut self.history {
            h.fill(0.0);
        }
    }

    /// Get the current history buffer state
    ///
    /// Returns ALL frames currently held in the ring buffer, exactly as they
    /// exist. Length is in the range `0..=kernel_size`. In particular, the
    /// returned buffer is enough to fully reproduce the convolution state for a
    /// round trip with [`Self::set_history`].
    ///
    /// Note: under normal usage, `history.len()` is either `kernel_size - 1`
    /// (between calls) or `kernel_size` (mid-`forward_step`, before trim). The
    /// fresh state has `kernel_size - 1` zero-filled frames.
    pub fn get_history(&self) -> Vec<Vec<f32>> {
        self.history.clone()
    }

    /// Set the history buffer state
    ///
    /// Accepts a history buffer of any length up to `kernel_size`. Each frame
    /// must have exactly `in_channels` elements. Returns
    /// [`CoreError::DimensionMismatch`] if either constraint is violated, so
    /// the caller can surface state-restoration failures rather than panicking.
    pub fn set_history(&mut self, history: Vec<Vec<f32>>) -> CoreResult<()> {
        if history.len() > self.kernel_size {
            return Err(CoreError::DimensionMismatch {
                expected: self.kernel_size,
                got: history.len(),
            });
        }
        for h in &history {
            if h.len() != self.in_channels {
                return Err(CoreError::DimensionMismatch {
                    expected: self.in_channels,
                    got: h.len(),
                });
            }
        }
        self.history = history;
        Ok(())
    }

    /// Get kernel size
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Get input channels
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Get output channels
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
}

/// Depthwise Causal Convolution (used in Mamba)
///
/// Each input channel has its own kernel (groups = in_channels).
/// More efficient than standard convolution for SSM preprocessing.
#[derive(Debug, Clone)]
pub struct DepthwiseCausalConv1d {
    /// Kernel weights [channels, kernel_size]
    weights: Vec<Vec<f32>>,
    /// Bias per channel
    bias: Vec<f32>,
    /// Kernel size
    kernel_size: usize,
    /// Number of channels
    channels: usize,
    /// Ring buffer for causal history [kernel_size - 1, channels]
    history: Vec<Vec<f32>>,
}

impl DepthwiseCausalConv1d {
    /// Create a new depthwise causal convolution
    pub fn new(channels: usize, kernel_size: usize) -> Self {
        let scale = (2.0 / kernel_size as f32).sqrt();
        let weights: Vec<Vec<f32>> = (0..channels)
            .map(|c| {
                (0..kernel_size)
                    .map(|k| ((c + k) as f32 * 0.1).sin() * scale)
                    .collect()
            })
            .collect();

        let bias = vec![0.0; channels];
        let history: Vec<Vec<f32>> = (0..(kernel_size - 1))
            .map(|_| vec![0.0; channels])
            .collect();

        Self {
            weights,
            bias,
            kernel_size,
            channels,
            history,
        }
    }

    /// Set weights.
    ///
    /// # Panics
    /// Panics if `weights` does not match `[channels, kernel_size]`. Use
    /// [`Self::set_weights_checked`] for a non-panicking variant.
    pub fn set_weights(&mut self, weights: Vec<Vec<f32>>) {
        self.set_weights_checked(weights)
            .unwrap_or_else(|e| panic!("DepthwiseCausalConv1d::set_weights: {e}"));
    }

    /// Set weights, returning [`CoreError::DimensionMismatch`] on a shape
    /// mismatch instead of panicking.
    pub fn set_weights_checked(&mut self, weights: Vec<Vec<f32>>) -> CoreResult<()> {
        if weights.len() != self.channels {
            return Err(CoreError::DimensionMismatch {
                expected: self.channels,
                got: weights.len(),
            });
        }
        for w in &weights {
            if w.len() != self.kernel_size {
                return Err(CoreError::DimensionMismatch {
                    expected: self.kernel_size,
                    got: w.len(),
                });
            }
        }
        self.weights = weights;
        Ok(())
    }

    /// Set bias.
    ///
    /// # Panics
    /// Panics if `bias.len() != channels`. Use [`Self::set_bias_checked`]
    /// for a non-panicking variant.
    pub fn set_bias(&mut self, bias: Vec<f32>) {
        self.set_bias_checked(bias)
            .unwrap_or_else(|e| panic!("DepthwiseCausalConv1d::set_bias: {e}"));
    }

    /// Set bias, returning [`CoreError::DimensionMismatch`] on a length
    /// mismatch instead of panicking.
    pub fn set_bias_checked(&mut self, bias: Vec<f32>) -> CoreResult<()> {
        if bias.len() != self.channels {
            return Err(CoreError::DimensionMismatch {
                expected: self.channels,
                got: bias.len(),
            });
        }
        self.bias = bias;
        Ok(())
    }

    /// Forward pass for single time step.
    ///
    /// # Panics
    /// Panics if `input.len() != channels`. Use
    /// [`Self::forward_step_checked`] for a non-panicking variant.
    pub fn forward_step(&mut self, input: &[f32]) -> Vec<f32> {
        self.forward_step_checked(input)
            .unwrap_or_else(|e| panic!("DepthwiseCausalConv1d::forward_step: {e}"))
    }

    /// Forward pass for single time step, returning
    /// [`CoreError::DimensionMismatch`] if `input.len() != channels` instead
    /// of panicking.
    pub fn forward_step_checked(&mut self, input: &[f32]) -> CoreResult<Vec<f32>> {
        if input.len() != self.channels {
            return Err(CoreError::DimensionMismatch {
                expected: self.channels,
                got: input.len(),
            });
        }

        self.history.push(input.to_vec());
        while self.history.len() > self.kernel_size {
            self.history.remove(0);
        }

        let mut output = self.bias.clone();

        for (c, kernel) in self.weights.iter().enumerate() {
            for (k, &weight) in kernel.iter().enumerate() {
                if k < self.history.len() {
                    let hist_idx = self.history.len() - 1 - k;
                    output[c] += weight * self.history[hist_idx][c];
                }
            }
        }

        Ok(output)
    }

    /// Forward for Array1.
    ///
    /// # Panics
    /// Panics if `input.len() != channels`. Handles non-contiguous `input`
    /// views correctly (via `ArrayBase::as_standard_layout`) rather than
    /// assuming contiguity.
    pub fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        // Copy through `as_standard_layout()` -> `.iter()` instead of the
        // previous `input.as_slice().expect("invariant: Array1 is
        // contiguous")`: that "invariant" does not actually hold for a
        // caller, since a non-contiguous owned `Array1` (e.g. produced by
        // `slice_move(s![..;2])`) makes `as_slice()` return `None` and
        // panic. Iterating handles any layout without assuming contiguity.
        let contiguous: Vec<f32> = input.as_standard_layout().iter().copied().collect();
        Array1::from_vec(self.forward_step(&contiguous))
    }

    /// Forward pass for batch
    pub fn forward_batch(&mut self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        input.iter().map(|x| self.forward_step(x)).collect()
    }

    /// Reset history
    pub fn reset(&mut self) {
        for h in &mut self.history {
            h.fill(0.0);
        }
    }

    /// Get the current history buffer state
    ///
    /// Returns ALL frames currently held in the ring buffer, exactly as they
    /// exist. Length is in the range `0..=kernel_size`. The fresh state has
    /// `kernel_size - 1` zero-filled frames; after each `forward_step`, the
    /// buffer is trimmed back to `kernel_size - 1`. Used for full-fidelity
    /// state snapshots.
    pub fn get_history(&self) -> Vec<Vec<f32>> {
        self.history.clone()
    }

    /// Set the history buffer state
    ///
    /// Accepts a history buffer of any length up to `kernel_size`. Each frame
    /// must have exactly `channels` elements. Returns
    /// [`CoreError::DimensionMismatch`] otherwise.
    pub fn set_history(&mut self, history: Vec<Vec<f32>>) -> CoreResult<()> {
        if history.len() > self.kernel_size {
            return Err(CoreError::DimensionMismatch {
                expected: self.kernel_size,
                got: history.len(),
            });
        }
        for h in &history {
            if h.len() != self.channels {
                return Err(CoreError::DimensionMismatch {
                    expected: self.channels,
                    got: h.len(),
                });
            }
        }
        self.history = history;
        Ok(())
    }

    /// Get kernel size
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Get channels
    pub fn channels(&self) -> usize {
        self.channels
    }
}

/// Short convolution for SSM (commonly kernel_size=4 in Mamba)
///
/// Optimized implementation for small kernel sizes using loop unrolling.
#[derive(Debug, Clone)]
pub struct ShortConv {
    /// The underlying depthwise convolution
    conv: DepthwiseCausalConv1d,
}

impl ShortConv {
    /// Create a new short convolution (defaults to kernel_size=4)
    pub fn new(channels: usize) -> Self {
        Self::with_kernel_size(channels, 4)
    }

    /// Create with custom kernel size
    pub fn with_kernel_size(channels: usize, kernel_size: usize) -> Self {
        Self {
            conv: DepthwiseCausalConv1d::new(channels, kernel_size),
        }
    }

    /// Forward pass
    pub fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        self.conv.forward(input)
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.conv.reset();
    }

    /// Set weights.
    ///
    /// # Panics
    /// Panics if `weights` does not match `[channels, kernel_size]`. Use
    /// [`Self::set_weights_checked`] for a non-panicking variant.
    pub fn set_weights(&mut self, weights: Vec<Vec<f32>>) {
        self.conv.set_weights(weights);
    }

    /// Set weights, returning [`CoreError::DimensionMismatch`] on a shape
    /// mismatch instead of panicking.
    pub fn set_weights_checked(&mut self, weights: Vec<Vec<f32>>) -> CoreResult<()> {
        self.conv.set_weights_checked(weights)
    }

    /// Get channels
    pub fn channels(&self) -> usize {
        self.conv.channels()
    }
}

/// Dilated Causal Convolution
///
/// Supports dilation for increasing receptive field without
/// increasing kernel size or computation.
#[derive(Debug, Clone)]
pub struct DilatedCausalConv1d {
    /// Kernel weights [channels, kernel_size]
    weights: Vec<Vec<f32>>,
    /// Bias
    bias: Vec<f32>,
    /// Kernel size
    kernel_size: usize,
    /// Dilation factor
    dilation: usize,
    /// Channels
    channels: usize,
    /// History buffer [effective_kernel_size, channels]
    history: Vec<Vec<f32>>,
}

impl DilatedCausalConv1d {
    /// Create a new dilated causal convolution
    pub fn new(channels: usize, kernel_size: usize, dilation: usize) -> Self {
        let scale = (2.0 / kernel_size as f32).sqrt();
        let weights: Vec<Vec<f32>> = (0..channels)
            .map(|c| {
                (0..kernel_size)
                    .map(|k| ((c + k) as f32 * 0.1).sin() * scale)
                    .collect()
            })
            .collect();

        let bias = vec![0.0; channels];

        // Effective kernel size for history: (kernel_size - 1) * dilation + 1
        let effective_size = (kernel_size - 1) * dilation;
        let history: Vec<Vec<f32>> = (0..effective_size).map(|_| vec![0.0; channels]).collect();

        Self {
            weights,
            bias,
            kernel_size,
            dilation,
            channels,
            history,
        }
    }

    /// Forward pass for single time step.
    ///
    /// # Panics
    /// Panics if `input.len() != channels`. Use
    /// [`Self::forward_step_checked`] for a non-panicking variant.
    pub fn forward_step(&mut self, input: &[f32]) -> Vec<f32> {
        self.forward_step_checked(input)
            .unwrap_or_else(|e| panic!("DilatedCausalConv1d::forward_step: {e}"))
    }

    /// Forward pass for single time step, returning
    /// [`CoreError::DimensionMismatch`] if `input.len() != channels` instead
    /// of panicking.
    pub fn forward_step_checked(&mut self, input: &[f32]) -> CoreResult<Vec<f32>> {
        if input.len() != self.channels {
            return Err(CoreError::DimensionMismatch {
                expected: self.channels,
                got: input.len(),
            });
        }

        self.history.push(input.to_vec());
        let effective_size = (self.kernel_size - 1) * self.dilation;
        while self.history.len() > effective_size + 1 {
            self.history.remove(0);
        }

        let mut output = self.bias.clone();

        for (c, kernel) in self.weights.iter().enumerate() {
            for (k, &weight) in kernel.iter().enumerate() {
                // Dilated index: current is at end, go back by k * dilation
                let offset = k * self.dilation;
                if offset < self.history.len() {
                    let hist_idx = self.history.len() - 1 - offset;
                    output[c] += weight * self.history[hist_idx][c];
                }
            }
        }

        Ok(output)
    }

    /// Forward for Array1.
    ///
    /// # Panics
    /// Panics if `input.len() != channels`. Handles non-contiguous `input`
    /// views correctly (via `ArrayBase::as_standard_layout`) rather than
    /// assuming contiguity.
    pub fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        // See `DepthwiseCausalConv1d::forward` for why this does not use
        // `input.as_slice().expect(...)`.
        let contiguous: Vec<f32> = input.as_standard_layout().iter().copied().collect();
        Array1::from_vec(self.forward_step(&contiguous))
    }

    /// Reset history
    pub fn reset(&mut self) {
        for h in &mut self.history {
            h.fill(0.0);
        }
    }

    /// Get receptive field
    pub fn receptive_field(&self) -> usize {
        (self.kernel_size - 1) * self.dilation + 1
    }
}

/// Stack of dilated causal convolutions (WaveNet-style)
///
/// Each layer has increasing dilation: 1, 2, 4, 8, ...
#[derive(Debug, Clone)]
pub struct DilatedStack {
    layers: Vec<DilatedCausalConv1d>,
    residual: bool,
}

impl DilatedStack {
    /// Create a new dilated stack with num_layers
    ///
    /// Dilations: 2^0, 2^1, 2^2, ..., 2^(num_layers-1)
    pub fn new(channels: usize, kernel_size: usize, num_layers: usize) -> Self {
        let layers: Vec<_> = (0..num_layers)
            .map(|i| {
                let dilation = 1 << i; // 2^i
                DilatedCausalConv1d::new(channels, kernel_size, dilation)
            })
            .collect();

        Self {
            layers,
            residual: true,
        }
    }

    /// Disable residual connections
    pub fn without_residual(mut self) -> Self {
        self.residual = false;
        self
    }

    /// Forward pass
    pub fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let mut x = input.clone();
        for layer in &mut self.layers {
            let y = layer.forward(&x);
            if self.residual {
                x = &x + &y;
            } else {
                x = y;
            }
        }
        x
    }

    /// Reset all layers
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Get total receptive field
    pub fn receptive_field(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.receptive_field() - 1)
            .sum::<usize>()
            + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_conv1d() {
        let mut conv = CausalConv1d::new(2, 3, 3);

        // First step - only current input contributes
        let out1 = conv.forward_step(&[1.0, 0.0]);
        assert_eq!(out1.len(), 3);

        // Second step - current + previous
        let out2 = conv.forward_step(&[0.0, 1.0]);
        assert_eq!(out2.len(), 3);

        // Third step - full kernel used
        let out3 = conv.forward_step(&[0.5, 0.5]);
        assert_eq!(out3.len(), 3);
    }

    #[test]
    fn test_depthwise_causal() {
        let mut conv = DepthwiseCausalConv1d::new(4, 3);

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let out = conv.forward(&input);
        assert_eq!(out.len(), 4);

        // After reset, should behave as if fresh
        conv.reset();
        let out2 = conv.forward(&input);
        assert_eq!(out, out2);
    }

    #[test]
    fn test_short_conv() {
        let mut conv = ShortConv::new(8);
        assert_eq!(conv.channels(), 8);

        let input = Array1::ones(8);
        let out = conv.forward(&input);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_dilated_conv() {
        let mut conv = DilatedCausalConv1d::new(4, 3, 2);
        assert_eq!(conv.receptive_field(), 5); // (3-1)*2 + 1

        let input = Array1::ones(4);
        let out = conv.forward(&input);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_dilated_stack() {
        let mut stack = DilatedStack::new(4, 2, 4);
        // Receptive field: 1 + 2 + 4 + 8 = 15
        // Actually: layers have dilations 1,2,4,8 with kernel_size=2
        // RF = sum((k-1)*d) + 1 = (1*1) + (1*2) + (1*4) + (1*8) + 1 = 16

        let input = Array1::ones(4);
        let out = stack.forward(&input);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_causality() {
        // Verify that output only depends on current and past inputs
        let mut conv1 = DepthwiseCausalConv1d::new(2, 3);
        let mut conv2 = DepthwiseCausalConv1d::new(2, 3);

        // Same weights
        conv2.set_weights(conv1.weights.clone());
        conv2.set_bias(conv1.bias.clone());

        // Feed same first two inputs
        let in1 = vec![1.0, 0.0];
        let in2 = vec![0.0, 1.0];

        let _ = conv1.forward_step(&in1);
        let out1 = conv1.forward_step(&in2);

        let _ = conv2.forward_step(&in1);
        let out2 = conv2.forward_step(&in2);

        // Outputs should be identical (causality preserved)
        assert_eq!(out1, out2);

        // Now feed different third inputs - previous outputs should have been same
        let _ = conv1.forward_step(&[1.0, 1.0]);
        let _ = conv2.forward_step(&[0.5, 0.5]);

        // First two outputs were identical, proving causality
    }

    /// Generate a deterministic test sequence for round-trip tests.
    fn make_sequence(num_steps: usize, channels: usize) -> Vec<Vec<f32>> {
        (0..num_steps)
            .map(|t| {
                (0..channels)
                    .map(|c| 0.05 + (t as f32) * 0.07 + (c as f32) * 0.013)
                    .collect()
            })
            .collect()
    }

    /// Round-trip check helper for `CausalConv1d`.
    ///
    /// Drives the conv `num_steps` times with a deterministic sequence,
    /// snapshots its state, advances it with a "throwaway" step that is
    /// guaranteed to mutate the buffer, then restores the snapshot and steps
    /// it forward with the SAME final input. The post-snapshot output must
    /// match the snapshot-restored output bit-exactly: if `get_history` /
    /// `set_history` are lossy, the two outputs diverge.
    fn assert_causal_conv1d_roundtrip(in_channels: usize, kernel_size: usize, num_steps: usize) {
        let mut conv = CausalConv1d::new(in_channels, in_channels.max(1), kernel_size);
        let sequence = make_sequence(num_steps + 2, in_channels);

        // Warm up `num_steps`.
        for step in sequence.iter().take(num_steps) {
            let _ = conv.forward_step(step);
        }

        // Snapshot, then step with sequence[num_steps] and record the output.
        let snapshot = conv.get_history();
        let final_input = &sequence[num_steps];
        let original_output = conv.forward_step(final_input);

        // Mutate the conv further so any state leakage shows up.
        for step in sequence.iter().skip(num_steps + 1) {
            let _ = conv.forward_step(step);
        }

        // Restore snapshot and replay the SAME step. Output must match exactly.
        conv.set_history(snapshot)
            .expect("set_history must accept its own snapshot");
        let restored_output = conv.forward_step(final_input);

        assert_eq!(
            original_output.len(),
            restored_output.len(),
            "round-trip output length mismatch at num_steps={num_steps}",
        );
        for (i, (a, b)) in original_output
            .iter()
            .zip(restored_output.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-6,
                "Causal conv round-trip differ at num_steps={num_steps}, idx {i}: {a} vs {b}",
            );
        }
    }

    /// Round-trip check helper for `DepthwiseCausalConv1d`.
    fn assert_depthwise_roundtrip(channels: usize, kernel_size: usize, num_steps: usize) {
        let mut conv = DepthwiseCausalConv1d::new(channels, kernel_size);
        let sequence = make_sequence(num_steps + 2, channels);

        for step in sequence.iter().take(num_steps) {
            let _ = conv.forward_step(step);
        }

        let snapshot = conv.get_history();
        let final_input = &sequence[num_steps];
        let original_output = conv.forward_step(final_input);

        for step in sequence.iter().skip(num_steps + 1) {
            let _ = conv.forward_step(step);
        }

        conv.set_history(snapshot)
            .expect("set_history must accept its own snapshot");
        let restored_output = conv.forward_step(final_input);

        for (i, (a, b)) in original_output
            .iter()
            .zip(restored_output.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-6,
                "Depthwise conv round-trip differ at num_steps={num_steps}, idx {i}: {a} vs {b}",
            );
        }
    }

    #[test]
    fn test_causal_conv1d_history_roundtrip() {
        // Cover the boundary cases: no steps (initial buffer of zeros),
        // partial fill, exactly kernel_size - 1, exactly kernel_size, and
        // well past kernel_size so the ring buffer has been trimmed.
        let kernel_size = 4;
        for num_steps in [0, 1, kernel_size - 1, kernel_size, kernel_size + 5] {
            assert_causal_conv1d_roundtrip(3, kernel_size, num_steps);
        }
    }

    #[test]
    fn test_depthwise_causal_history_roundtrip() {
        let kernel_size = 5;
        for num_steps in [0, 1, kernel_size - 1, kernel_size, kernel_size + 7] {
            assert_depthwise_roundtrip(4, kernel_size, num_steps);
        }
    }

    #[test]
    fn test_causal_conv1d_set_history_rejects_oversized() {
        let mut conv = CausalConv1d::new(2, 2, 3);
        // kernel_size = 3, so anything > 3 frames must be rejected.
        let too_many = vec![vec![0.0; 2]; 4];
        assert!(conv.set_history(too_many).is_err());

        // Wrong channel width must also be rejected.
        let wrong_width = vec![vec![0.0; 7]; 2];
        assert!(conv.set_history(wrong_width).is_err());
    }

    #[test]
    fn test_depthwise_set_history_rejects_oversized() {
        let mut conv = DepthwiseCausalConv1d::new(4, 3);
        let too_many = vec![vec![0.0; 4]; 5];
        assert!(conv.set_history(too_many).is_err());

        let wrong_width = vec![vec![0.0; 9]; 2];
        assert!(conv.set_history(wrong_width).is_err());
    }

    // ------------------------------------------------------------------
    // Regression tests: checked (non-panicking) variants and non-contiguous
    // `forward()` input.
    // ------------------------------------------------------------------

    #[test]
    fn test_causal_conv1d_checked_methods_reject_bad_dimensions() {
        let mut conv = CausalConv1d::new(2, 3, 3); // in=2, out=3, kernel=3

        // set_weights_checked: wrong out_channels count.
        assert!(conv
            .set_weights_checked(vec![vec![vec![0.0; 3]; 2]; 5])
            .is_err());
        // set_weights_checked: wrong in_channels count (inner Vec length).
        assert!(conv
            .set_weights_checked(vec![vec![vec![0.0; 3]; 7]; 3])
            .is_err());
        // set_weights_checked: wrong kernel_size (innermost Vec length).
        assert!(conv
            .set_weights_checked(vec![vec![vec![0.0; 9]; 2]; 3])
            .is_err());
        // Valid shape must succeed.
        assert!(conv
            .set_weights_checked(vec![vec![vec![0.1; 3]; 2]; 3])
            .is_ok());

        // set_bias_checked: wrong length.
        assert!(conv.set_bias_checked(vec![0.0; 7]).is_err());
        assert!(conv.set_bias_checked(vec![0.0; 3]).is_ok());

        // forward_step_checked: wrong input length.
        assert!(conv.forward_step_checked(&[1.0, 2.0, 3.0]).is_err());
        assert!(conv.forward_step_checked(&[1.0, 2.0]).is_ok());
    }

    #[test]
    #[should_panic(expected = "Dimension mismatch")]
    fn test_causal_conv1d_set_weights_still_panics_on_bad_shape() {
        // The infallible `set_weights` API is kept (external crates already
        // call it), but must still reject bad input loudly rather than
        // silently accepting a malformed shape.
        let mut conv = CausalConv1d::new(2, 3, 3);
        conv.set_weights(vec![vec![vec![0.0; 3]; 2]; 999]);
    }

    #[test]
    fn test_depthwise_conv1d_checked_methods_reject_bad_dimensions() {
        let mut conv = DepthwiseCausalConv1d::new(4, 3);

        assert!(conv.set_weights_checked(vec![vec![0.0; 3]; 9]).is_err());
        assert!(conv.set_weights_checked(vec![vec![0.0; 9]; 4]).is_err());
        assert!(conv.set_weights_checked(vec![vec![0.1; 3]; 4]).is_ok());

        assert!(conv.set_bias_checked(vec![0.0; 9]).is_err());
        assert!(conv.set_bias_checked(vec![0.0; 4]).is_ok());

        assert!(conv.forward_step_checked(&[1.0, 2.0]).is_err());
        assert!(conv.forward_step_checked(&[1.0, 2.0, 3.0, 4.0]).is_ok());
    }

    #[test]
    fn test_depthwise_forward_handles_non_contiguous_array1() {
        // Regression: `forward()` used to do
        // `input.as_slice().expect("invariant: Array1 is contiguous")`,
        // which panics for a non-contiguous owned `Array1` -- an invariant
        // that does NOT actually hold for arbitrary callers.
        let mut conv = DepthwiseCausalConv1d::new(3, 2);

        // Build a genuinely non-contiguous owned Array1 by slicing with a
        // stride and then materialising it back as an owned array via
        // `to_owned()` on the strided view -- `slice` on an Array1 keeps the
        // stride, so `.to_owned()` here still yields distinct data but we
        // instead construct non-contiguity directly via `slice_move`.
        let source = Array1::from_vec(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
        let strided = source.slice_move(scirs2_core::ndarray::s![..;2]); // [1.0, 2.0, 3.0], stride 2
        assert!(
            strided.as_slice().is_none(),
            "test precondition: input must be non-contiguous"
        );
        assert_eq!(strided.len(), 3);

        // Must not panic, and must process the actual (strided) values.
        let out = conv.forward(&strided);
        assert_eq!(out.len(), 3);

        // Cross-check against the equivalent contiguous input.
        let mut conv_ref = DepthwiseCausalConv1d::new(3, 2);
        conv_ref.set_weights(conv.weights.clone());
        conv_ref.set_bias(conv.bias.clone());
        let contiguous = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let out_ref = conv_ref.forward(&contiguous);

        for i in 0..3 {
            assert!(
                (out[i] - out_ref[i]).abs() < 1e-6,
                "index {i}: strided-input forward {} != contiguous-input forward {}",
                out[i],
                out_ref[i]
            );
        }
    }

    #[test]
    fn test_dilated_conv1d_checked_forward_step_rejects_bad_length() {
        let mut conv = DilatedCausalConv1d::new(4, 3, 2);
        assert!(conv.forward_step_checked(&[1.0, 2.0]).is_err());
        assert!(conv.forward_step_checked(&[1.0, 2.0, 3.0, 4.0]).is_ok());
    }
}
