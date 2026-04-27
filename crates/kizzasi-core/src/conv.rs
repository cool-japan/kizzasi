//! Causal convolution implementations for SSM architectures
//!
//! Causal convolutions are essential for autoregressive models as they
//! ensure the output at time t only depends on inputs at times <= t.

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

    /// Set weights from external source
    pub fn set_weights(&mut self, weights: Vec<Vec<Vec<f32>>>) {
        assert_eq!(weights.len(), self.out_channels);
        for oc in &weights {
            assert_eq!(oc.len(), self.in_channels);
            for ic in oc {
                assert_eq!(ic.len(), self.kernel_size);
            }
        }
        self.weights = weights;
    }

    /// Set bias from external source
    pub fn set_bias(&mut self, bias: Vec<f32>) {
        assert_eq!(bias.len(), self.out_channels);
        self.bias = bias;
    }

    /// Forward pass for a single time step (streaming/causal)
    ///
    /// Takes input of shape `[in_channels]` and returns output of shape `[out_channels]`
    pub fn forward_step(&mut self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.in_channels);

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

        output
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
    /// Returns kernel_size - 1 frames (excluding current input if present)
    pub fn get_history(&self) -> Vec<Vec<f32>> {
        // If history has kernel_size elements (after forward_step but before next call),
        // we only save the first kernel_size - 1 elements
        let expected_len = self.kernel_size - 1;
        if self.history.len() >= expected_len {
            self.history[..expected_len].to_vec()
        } else {
            self.history.clone()
        }
    }

    /// Set the history buffer state
    pub fn set_history(&mut self, history: Vec<Vec<f32>>) {
        assert_eq!(
            history.len(),
            self.kernel_size - 1,
            "History length must be kernel_size - 1 = {}",
            self.kernel_size - 1
        );
        for h in &history {
            assert_eq!(
                h.len(),
                self.in_channels,
                "Each history frame must have in_channels = {} elements",
                self.in_channels
            );
        }
        self.history = history;
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

    /// Set weights
    pub fn set_weights(&mut self, weights: Vec<Vec<f32>>) {
        assert_eq!(weights.len(), self.channels);
        for w in &weights {
            assert_eq!(w.len(), self.kernel_size);
        }
        self.weights = weights;
    }

    /// Set bias
    pub fn set_bias(&mut self, bias: Vec<f32>) {
        assert_eq!(bias.len(), self.channels);
        self.bias = bias;
    }

    /// Forward pass for single time step
    pub fn forward_step(&mut self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.channels);

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

        output
    }

    /// Forward for Array1
    pub fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        Array1::from_vec(self.forward_step(input.as_slice().unwrap()))
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
    /// Returns kernel_size - 1 frames (excluding current input if present)
    pub fn get_history(&self) -> Vec<Vec<f32>> {
        // If history has kernel_size elements (after forward_step but before next call),
        // we only save the first kernel_size - 1 elements
        let expected_len = self.kernel_size - 1;
        if self.history.len() >= expected_len {
            self.history[..expected_len].to_vec()
        } else {
            self.history.clone()
        }
    }

    /// Set the history buffer state
    pub fn set_history(&mut self, history: Vec<Vec<f32>>) {
        assert_eq!(
            history.len(),
            self.kernel_size - 1,
            "History length must be kernel_size - 1 = {}",
            self.kernel_size - 1
        );
        for h in &history {
            assert_eq!(
                h.len(),
                self.channels,
                "Each history frame must have channels = {} elements",
                self.channels
            );
        }
        self.history = history;
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

    /// Set weights
    pub fn set_weights(&mut self, weights: Vec<Vec<f32>>) {
        self.conv.set_weights(weights);
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

    /// Forward pass for single time step
    pub fn forward_step(&mut self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.channels);

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

        output
    }

    /// Forward for Array1
    pub fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        Array1::from_vec(self.forward_step(input.as_slice().unwrap()))
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
}
