//! Neuromorphic Spiking Neural Networks (SNN)
//!
//! This module implements biologically-inspired Spiking Neural Networks using
//! the Leaky Integrate-and-Fire (LIF) neuron model. SNNs encode information in
//! the timing and rate of discrete spikes rather than continuous activations.
//!
//! # Architecture
//!
//! ```text
//! Input → [LIF Layer 1] → spikes → [LIF Layer 2] → spikes → ... → [Output Projection]
//!                ↑ membrane state                        ↑ membrane state
//! ```
//!
//! # LIF Neuron Dynamics
//!
//! ```text
//! V(t+1) = leak_factor * V(t) + W @ input + bias
//! spike(t) = 1 if V(t+1) >= threshold AND refractory_countdown == 0
//! ```
//!
//! After a spike, the reset mode determines the new membrane potential:
//! - `HardReset`: V → 0
//! - `SoftReset`: V → V - threshold
//! - `SubThreshold`: V → V * leak_factor
//!
//! # STDP Learning
//!
//! Spike-Timing Dependent Plasticity (STDP) is a biologically-plausible
//! unsupervised learning rule that adjusts synaptic weights based on the
//! relative timing of pre- and post-synaptic spikes.
//!
//! # References
//!
//! - "Leaky integrate-and-fire model" (Dayan & Abbott, 2001)
//! - "Spike-timing-dependent plasticity" (Bi & Poo, 1998)

use crate::error::{ModelError, ModelResult};
use crate::{AutoregressiveModel, ModelType};
use kizzasi_core::{CoreResult, HiddenState, SignalPredictor};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use tracing::{debug, instrument, trace};

// ---------------------------------------------------------------------------
// Seeded deterministic RNG for reproducible weight initialization
// ---------------------------------------------------------------------------

/// Simple xorshift64 PRNG for deterministic weight initialization.
struct SeededRng {
    state: u64,
}

impl SeededRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    /// Returns a float in [-1, 1)
    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
    }
}

// ---------------------------------------------------------------------------
// Reset Mode
// ---------------------------------------------------------------------------

/// What happens to membrane potential after a spike
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResetMode {
    /// Membrane potential resets to 0 after spike
    HardReset,
    /// Membrane potential is reduced by threshold: V → V - threshold
    SoftReset,
    /// Membrane potential decays by leak factor: V → V * leak_factor
    SubThreshold,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Spiking Neural Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension (neurons per hidden layer)
    pub hidden_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Number of hidden LIF layers
    pub num_layers: usize,
    /// Membrane voltage threshold for spiking (default: 1.0)
    pub threshold: f32,
    /// Membrane decay factor per step (default: 0.9)
    pub leak_factor: f32,
    /// Steps a neuron must wait after spiking before it can spike again (default: 2)
    pub refractory_period: usize,
    /// What happens to membrane potential after a spike
    pub reset_mode: ResetMode,
    /// Time step in ms (default: 1.0)
    pub dt: f32,
}

impl SpikingConfig {
    /// Create a default SpikingConfig
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, num_layers: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            threshold: 1.0,
            leak_factor: 0.9,
            refractory_period: 2,
            reset_mode: ResetMode::SoftReset,
            dt: 1.0,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> ModelResult<()> {
        if self.input_dim == 0 {
            return Err(ModelError::invalid_config("input_dim must be > 0"));
        }
        if self.hidden_dim == 0 {
            return Err(ModelError::invalid_config("hidden_dim must be > 0"));
        }
        if self.output_dim == 0 {
            return Err(ModelError::invalid_config("output_dim must be > 0"));
        }
        if self.num_layers == 0 {
            return Err(ModelError::invalid_config("num_layers must be > 0"));
        }
        if !self.threshold.is_finite() || self.threshold <= 0.0 {
            return Err(ModelError::invalid_config(
                "threshold must be positive and finite",
            ));
        }
        if !(0.0..=1.0).contains(&self.leak_factor) {
            return Err(ModelError::invalid_config(
                "leak_factor must be in [0.0, 1.0]",
            ));
        }
        if self.dt <= 0.0 {
            return Err(ModelError::invalid_config("dt must be > 0"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Membrane Potential State
// ---------------------------------------------------------------------------

/// Membrane state for a layer of LIF neurons
#[derive(Debug, Clone)]
pub struct MembranePotential {
    /// Current membrane potential per neuron
    pub voltages: Array1<f32>,
    /// Steps remaining in refractory period per neuron
    pub refractory_countdown: Array1<u32>,
    /// Eligibility trace for STDP (decays over time)
    pub last_spike_trace: Array1<f32>,
}

impl MembranePotential {
    /// Create zeroed membrane state for `n` neurons
    pub fn new(n: usize) -> Self {
        Self {
            voltages: Array1::zeros(n),
            refractory_countdown: Array1::zeros(n),
            last_spike_trace: Array1::zeros(n),
        }
    }

    /// Reset all state to zero
    pub fn reset(&mut self) {
        self.voltages.fill(0.0);
        self.refractory_countdown.fill(0);
        self.last_spike_trace.fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// LIF Layer
// ---------------------------------------------------------------------------

/// Leaky Integrate-and-Fire (LIF) neuron layer
pub struct LifLayer {
    config: SpikingConfig,
    /// Synaptic weight matrix: (output_neurons, input_neurons)
    weights: Array2<f32>,
    /// Bias per neuron
    bias: Array1<f32>,
    /// Output neuron count (for clarity)
    output_neurons: usize,
}

impl LifLayer {
    /// Create a new LIF layer
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input dimension
    /// * `output_dim` - Output dimension (number of LIF neurons)
    /// * `config` - SNN configuration (threshold, leak_factor, etc.)
    pub fn new(input_dim: usize, output_dim: usize, config: &SpikingConfig) -> ModelResult<Self> {
        if input_dim == 0 || output_dim == 0 {
            return Err(ModelError::invalid_config(
                "LIF layer dimensions must be > 0",
            ));
        }

        // Xavier-like initialization scaled to prevent early saturation
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
        let mut rng =
            SeededRng::new(((input_dim + output_dim) as u64).wrapping_mul(6364136223846793005));

        let weights = Array2::from_shape_fn((output_dim, input_dim), |_| rng.next_f32() * scale);
        let bias = Array1::from_shape_fn(output_dim, |_| rng.next_f32() * 0.01);

        Ok(Self {
            config: config.clone(),
            weights,
            bias,
            output_neurons: output_dim,
        })
    }

    /// Compute one time step of LIF dynamics.
    ///
    /// Integrates inputs into membrane potential, checks threshold, emits spikes,
    /// and applies refractory period. Returns binary spike vector (0.0 or 1.0).
    #[instrument(skip(self, input, state), fields(neurons = self.output_neurons))]
    pub fn step(
        &self,
        input: &Array1<f32>,
        state: &mut MembranePotential,
    ) -> ModelResult<Array1<f32>> {
        if input.len() != self.weights.ncols() {
            return Err(ModelError::dimension_mismatch(
                "LIF input",
                self.weights.ncols(),
                input.len(),
            ));
        }
        if state.voltages.len() != self.output_neurons {
            return Err(ModelError::dimension_mismatch(
                "LIF state",
                self.output_neurons,
                state.voltages.len(),
            ));
        }

        // Compute synaptic current: I = W @ input + bias
        let synaptic_current = self.weights.dot(input) + &self.bias;

        // Integrate: V(t+1) = leak * V(t) + I(t)
        let new_voltages = &state.voltages * self.config.leak_factor + &synaptic_current;

        // Determine which neurons spike: above threshold AND not in refractory period
        let mut spikes = Array1::<f32>::zeros(self.output_neurons);
        let mut updated_voltages = new_voltages.clone();
        let mut updated_refractory = state.refractory_countdown.clone();
        let mut updated_traces = state.last_spike_trace.clone();

        // Decay eligibility traces: trace(t+1) = trace(t) * exp(-dt/tau_trace)
        // Use tau_trace = 20 ms as a biologically reasonable default
        let tau_trace = 20.0_f32;
        let trace_decay = (-self.config.dt / tau_trace).exp();
        updated_traces.mapv_inplace(|t| t * trace_decay);

        for i in 0..self.output_neurons {
            if updated_refractory[i] > 0 {
                // In absolute refractory period: decrement counter, no spike, and clamp the
                // membrane at the reset value that was written when the neuron last spiked.
                // state.voltages[i] already holds that reset value (set at spike time and
                // persisted at the end of that step). Re-applying leak or adding synaptic
                // current would both violate standard absolute-refractory clamping
                // (Gerstner & Kistler, "Spiking Neuron Models" §2.1;
                //  Dayan & Abbott, "Theoretical Neuroscience" Ch.1).
                updated_refractory[i] -= 1;
                updated_voltages[i] = state.voltages[i];
            } else if new_voltages[i] >= self.config.threshold {
                // Neuron fires!
                spikes[i] = 1.0;
                updated_refractory[i] = self.config.refractory_period as u32;
                updated_traces[i] += 1.0;

                // Apply reset mode
                updated_voltages[i] = match self.config.reset_mode {
                    ResetMode::HardReset => 0.0,
                    ResetMode::SoftReset => new_voltages[i] - self.config.threshold,
                    ResetMode::SubThreshold => new_voltages[i] * self.config.leak_factor,
                };
            }
        }

        state.voltages = updated_voltages;
        state.refractory_countdown = updated_refractory;
        state.last_spike_trace = updated_traces;

        Ok(spikes)
    }

    /// Initialize membrane state for this layer
    pub fn init_state(&self) -> MembranePotential {
        MembranePotential::new(self.output_neurons)
    }

    /// Get number of output neurons
    pub fn output_neurons(&self) -> usize {
        self.output_neurons
    }

    /// Get mutable reference to weights (for STDP updates)
    pub fn weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.weights
    }
}

// ---------------------------------------------------------------------------
// STDP Updater
// ---------------------------------------------------------------------------

/// Spike-Timing Dependent Plasticity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StdpConfig {
    /// Potentiation amplitude — how much weights increase on coincident spikes (default: 0.01)
    pub a_plus: f32,
    /// Depression amplitude — how much weights decrease on anti-causal spikes (default: 0.012)
    pub a_minus: f32,
    /// Pre-synaptic trace time constant in ms (default: 20.0)
    pub tau_plus: f32,
    /// Post-synaptic trace time constant in ms (default: 20.0)
    pub tau_minus: f32,
}

impl Default for StdpConfig {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.012,
            tau_plus: 20.0,
            tau_minus: 20.0,
        }
    }
}

/// Implements Spike-Timing Dependent Plasticity weight updates
pub struct StdpUpdater {
    config: StdpConfig,
}

impl StdpUpdater {
    /// Create a new STDP updater with the given configuration
    pub fn new(config: StdpConfig) -> Self {
        Self { config }
    }

    /// Update synaptic weights based on pre/post-synaptic spike traces.
    ///
    /// Potentiation: when a post-synaptic neuron fires shortly after pre-synaptic activity
    ///   → ΔW = A+ * pre_trace (outer product with post_spikes)
    ///
    /// Depression: when a pre-synaptic neuron fires shortly after post-synaptic activity
    ///   → ΔW = -A- * post_trace (outer product with pre_traces)
    #[instrument(skip(self, weights, pre_traces, post_spikes, post_traces))]
    pub fn update_weights(
        &self,
        weights: &mut Array2<f32>,
        pre_traces: &Array1<f32>,
        post_spikes: &Array1<f32>,
        post_traces: &Array1<f32>,
    ) -> ModelResult<()> {
        let (n_out, n_in) = weights.dim();

        if pre_traces.len() != n_in {
            return Err(ModelError::dimension_mismatch(
                "STDP pre_traces",
                n_in,
                pre_traces.len(),
            ));
        }
        if post_spikes.len() != n_out {
            return Err(ModelError::dimension_mismatch(
                "STDP post_spikes",
                n_out,
                post_spikes.len(),
            ));
        }
        if post_traces.len() != n_out {
            return Err(ModelError::dimension_mismatch(
                "STDP post_traces",
                n_out,
                post_traces.len(),
            ));
        }

        // Potentiation: ΔW_ij += A+ * post_spike_i * pre_trace_j
        for i in 0..n_out {
            if post_spikes[i] > 0.0 {
                for j in 0..n_in {
                    weights[[i, j]] += self.config.a_plus * pre_traces[j];
                }
            }
        }

        // Depression: ΔW_ij -= A- * post_trace_i * pre_spike_j
        // pre_traces are used as proxy for recent pre-synaptic activity
        for i in 0..n_out {
            for j in 0..n_in {
                // Pre-trace represents pre-synaptic activity level
                weights[[i, j]] -= self.config.a_minus * post_traces[i] * pre_traces[j];
            }
        }

        // Soft weight normalization to prevent unbounded growth
        weights.mapv_inplace(|w| w.clamp(-10.0, 10.0));

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SpikingNeuralNetwork
// ---------------------------------------------------------------------------

/// Complete multi-layer Spiking Neural Network
pub struct SpikingNeuralNetwork {
    /// Model configuration
    pub config: SpikingConfig,
    /// Hidden LIF layers
    layers: Vec<LifLayer>,
    /// Output projection: (output_dim, hidden_dim)
    output_proj: Array2<f32>,
    /// Output bias
    output_bias: Array1<f32>,
    /// Membrane states for each layer
    layer_states: Vec<MembranePotential>,
    /// Optional STDP updater for online learning
    stdp: Option<StdpUpdater>,
    /// Total steps processed (for rate statistics)
    step_count: usize,
    /// Accumulated spike counts per layer for rate computation
    spike_accumulator: Vec<Array1<f32>>,
}

impl SpikingNeuralNetwork {
    /// Create a new SNN with the given configuration
    #[instrument(skip(config), fields(layers = config.num_layers, hidden = config.hidden_dim))]
    pub fn new(config: SpikingConfig) -> ModelResult<Self> {
        config.validate()?;
        Self::build(config, None)
    }

    /// Create a new SNN with STDP learning enabled
    pub fn new_with_stdp(config: SpikingConfig, stdp: StdpConfig) -> ModelResult<Self> {
        config.validate()?;
        Self::build(config, Some(StdpUpdater::new(stdp)))
    }

    /// Small preset: 2 layers, 64 hidden neurons
    pub fn small() -> ModelResult<Self> {
        let config = SpikingConfig::new(1, 64, 1, 2);
        Self::new(config)
    }

    fn build(config: SpikingConfig, stdp: Option<StdpUpdater>) -> ModelResult<Self> {
        debug!(
            "Building SNN: layers={}, hidden={}",
            config.num_layers, config.hidden_dim
        );

        let mut layers = Vec::with_capacity(config.num_layers);

        // First layer: input_dim → hidden_dim
        layers.push(LifLayer::new(config.input_dim, config.hidden_dim, &config)?);

        // Remaining hidden layers: hidden_dim → hidden_dim
        for _ in 1..config.num_layers {
            layers.push(LifLayer::new(
                config.hidden_dim,
                config.hidden_dim,
                &config,
            )?);
        }

        // Output projection: hidden_dim → output_dim
        let scale = (2.0 / (config.hidden_dim + config.output_dim) as f32).sqrt();
        let mut rng = SeededRng::new(
            ((config.hidden_dim * 1000 + config.output_dim) as u64)
                .wrapping_mul(2862933555777941757),
        );
        let output_proj = Array2::from_shape_fn((config.output_dim, config.hidden_dim), |_| {
            rng.next_f32() * scale
        });
        let output_bias = Array1::from_shape_fn(config.output_dim, |_| rng.next_f32() * 0.01);

        let layer_states: Vec<MembranePotential> = layers.iter().map(|l| l.init_state()).collect();

        let spike_accumulator: Vec<Array1<f32>> = layers
            .iter()
            .map(|l| Array1::zeros(l.output_neurons()))
            .collect();

        debug!("SNN built with {} layers", layers.len());

        Ok(Self {
            config,
            layers,
            output_proj,
            output_bias,
            layer_states,
            stdp,
            step_count: 0,
            spike_accumulator,
        })
    }

    /// Initialize fresh membrane states for all layers
    pub fn init_layer_states(&self) -> Vec<MembranePotential> {
        self.layers.iter().map(|l| l.init_state()).collect()
    }

    /// Compute average firing rate per layer (spikes per step since last reset)
    pub fn average_firing_rate(&self) -> Vec<f32> {
        if self.step_count == 0 {
            return vec![0.0; self.layers.len()];
        }
        self.spike_accumulator
            .iter()
            .map(|acc| acc.mean().unwrap_or(0.0) / self.step_count as f32)
            .collect()
    }

    /// Internal forward pass: propagates input through all LIF layers,
    /// applies optional STDP, then projects last layer spikes to output.
    fn forward_step(&mut self, input: &Array1<f32>) -> ModelResult<Array1<f32>> {
        let mut current = input.clone();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let state = self.layer_states.get_mut(layer_idx).ok_or_else(|| {
                ModelError::not_initialized(format!("layer state {} missing", layer_idx))
            })?;

            let spikes = layer.step(&current, state)?;

            // Accumulate firing stats
            if let Some(acc) = self.spike_accumulator.get_mut(layer_idx) {
                *acc += &spikes;
            }

            current = spikes;
        }

        // If STDP is enabled, apply learning on the last layer
        if let Some(stdp_updater) = &self.stdp {
            if self.layers.len() >= 2 {
                // Use penultimate layer traces as pre-synaptic signal
                let pre_traces = self
                    .layer_states
                    .get(self.layers.len() - 2)
                    .map(|s| s.last_spike_trace.clone())
                    .unwrap_or_else(|| Array1::zeros(self.config.hidden_dim));

                let post_spikes = current.clone();
                let post_traces = self
                    .layer_states
                    .last()
                    .map(|s| s.last_spike_trace.clone())
                    .unwrap_or_else(|| Array1::zeros(self.config.hidden_dim));

                // We need mutable access to the last layer's weights
                let last_idx = self.layers.len() - 1;
                // Safety: we only borrow a single layer mutably
                let weights = self.layers[last_idx].weights_mut();
                stdp_updater.update_weights(weights, &pre_traces, &post_spikes, &post_traces)?;
            }
        }

        // Rate coding output: project spike vector through output projection
        let output = self.output_proj.dot(&current) + &self.output_bias;

        // Check numerical stability
        if output.iter().any(|v| !v.is_finite()) {
            return Err(ModelError::numerical_instability(
                "SNN output projection",
                "NaN or Inf detected",
            ));
        }

        self.step_count += 1;
        Ok(output)
    }
}

impl SignalPredictor for SpikingNeuralNetwork {
    #[instrument(skip(self, input))]
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        self.forward_step(input)
            .map_err(|e| kizzasi_core::CoreError::Generic(e.to_string()))
    }

    #[instrument(skip(self))]
    fn reset(&mut self) {
        debug!("Resetting SNN membrane states");
        for state in &mut self.layer_states {
            state.reset();
        }
        for acc in &mut self.spike_accumulator {
            acc.fill(0.0);
        }
        self.step_count = 0;
    }

    fn context_window(&self) -> usize {
        // SNNs have recurrent dynamics — effectively unbounded context
        usize::MAX
    }
}

impl AutoregressiveModel for SpikingNeuralNetwork {
    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    fn state_dim(&self) -> usize {
        // State dimension = total membrane voltages across all layers
        self.config.hidden_dim * self.config.num_layers
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn model_type(&self) -> ModelType {
        ModelType::Snn
    }

    fn get_states(&self) -> Vec<HiddenState> {
        self.layer_states
            .iter()
            .map(|state| {
                let dim = state.voltages.len();
                let state_2d = state
                    .voltages
                    .clone()
                    .insert_axis(scirs2_core::ndarray::Axis(0));
                let mut hidden = HiddenState::new(dim, 1);
                hidden.update(state_2d);
                hidden
            })
            .collect()
    }

    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()> {
        if states.len() != self.config.num_layers {
            return Err(ModelError::state_count_mismatch(
                "SNN",
                self.config.num_layers,
                states.len(),
            ));
        }
        for (layer_state, hidden) in self.layer_states.iter_mut().zip(states.iter()) {
            let state_2d = hidden.state();
            if state_2d.nrows() > 0 && state_2d.ncols() > 0 {
                layer_state.voltages = state_2d.row(0).to_owned();
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> SpikingConfig {
        SpikingConfig::new(4, 8, 4, 2)
    }

    // 1. test_lif_layer_creation
    #[test]
    fn test_lif_layer_creation() {
        let config = small_config();
        let layer = LifLayer::new(4, 8, &config);
        assert!(layer.is_ok());
        let layer = layer.expect("LIF layer creation failed");
        assert_eq!(layer.output_neurons(), 8);
    }

    // 2. test_lif_spike_generation
    #[test]
    fn test_lif_spike_generation() {
        let config = SpikingConfig {
            threshold: 0.1,   // Low threshold → easy to spike
            leak_factor: 1.0, // No decay
            refractory_period: 0,
            ..SpikingConfig::new(4, 8, 4, 1)
        };
        let layer = LifLayer::new(4, 8, &config).expect("LIF layer creation failed");
        let mut state = layer.init_state();

        // Large input should trigger spikes
        let input = Array1::from_vec(vec![10.0_f32; 4]);
        let spikes = layer.step(&input, &mut state).expect("step failed");

        // At least some neurons should have spiked
        let total_spikes: f32 = spikes.sum();
        assert!(
            total_spikes > 0.0,
            "expected spikes for large input, got {total_spikes}"
        );
    }

    // 3. test_lif_refractory_period
    #[test]
    fn test_lif_refractory_period() {
        let config = SpikingConfig {
            threshold: 0.1,
            leak_factor: 1.0,
            refractory_period: 5, // Long refractory
            reset_mode: ResetMode::SoftReset,
            ..SpikingConfig::new(4, 8, 4, 1)
        };
        let layer = LifLayer::new(4, 8, &config).expect("LIF layer creation failed");
        let mut state = layer.init_state();

        // Force spike on step 1
        let input = Array1::from_vec(vec![10.0_f32; 4]);
        let spikes1 = layer.step(&input, &mut state).expect("step 1 failed");
        let total_spikes1: f32 = spikes1.sum();

        // Step 2: neurons in refractory should not spike
        let spikes2 = layer.step(&input, &mut state).expect("step 2 failed");
        let total_spikes2: f32 = spikes2.sum();

        // Neurons that spiked on step 1 should not spike on step 2
        if total_spikes1 > 0.0 {
            // Refractory neurons that spiked in step 1 can't spike in step 2
            // total_spikes2 should be <= total_spikes1 (some may be in refractory)
            assert!(
                total_spikes2 <= total_spikes1 || total_spikes2 == 0.0,
                "refractory period not respected: step1={total_spikes1}, step2={total_spikes2}"
            );
        }
    }

    // 4. test_lif_membrane_decay
    #[test]
    fn test_lif_membrane_decay() {
        let config = SpikingConfig {
            threshold: 1000.0, // Very high threshold → never spike
            leak_factor: 0.5,
            refractory_period: 0,
            reset_mode: ResetMode::HardReset,
            ..SpikingConfig::new(4, 8, 4, 1)
        };
        let layer = LifLayer::new(4, 8, &config).expect("LIF layer creation failed");
        let mut state = layer.init_state();

        // Apply small input to build up voltage
        let input = Array1::from_vec(vec![0.1_f32; 4]);
        let _ = layer.step(&input, &mut state).expect("step 1 failed");
        let v1 = state.voltages.clone();

        // Zero input: voltages should decay
        let zero_input = Array1::zeros(4);
        let _ = layer.step(&zero_input, &mut state).expect("step 2 failed");
        let v2 = state.voltages.clone();

        // With leak_factor=0.5, voltages should be approximately half of v1
        let v1_norm: f32 = v1.iter().map(|x| x.abs()).sum();
        let v2_norm: f32 = v2.iter().map(|x| x.abs()).sum();
        if v1_norm > 1e-6 {
            assert!(
                v2_norm < v1_norm,
                "membrane voltages should decay, got v1={v1_norm}, v2={v2_norm}"
            );
        }
    }

    // 5. test_snn_forward
    #[test]
    fn test_snn_forward() {
        let config = small_config();
        let output_dim = config.output_dim;
        let mut model = SpikingNeuralNetwork::new(config).expect("SNN creation failed");

        let input = Array1::from_vec(vec![0.5_f32; 4]);
        let output = model.forward_step(&input).expect("forward pass failed");

        assert_eq!(output.len(), output_dim);
        assert!(
            output.iter().all(|v| v.is_finite()),
            "output must be finite"
        );
    }

    // 6. test_snn_signal_predictor
    #[test]
    fn test_snn_signal_predictor() {
        let config = small_config();
        let output_dim = config.output_dim;
        let mut model = SpikingNeuralNetwork::new(config).expect("SNN creation failed");

        let input = Array1::from_vec(vec![0.1_f32; 4]);
        let output = model.step(&input).expect("SignalPredictor::step failed");

        assert_eq!(output.len(), output_dim);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    // 7. test_snn_reset
    #[test]
    fn test_snn_reset() {
        let config = small_config();
        let mut model = SpikingNeuralNetwork::new(config).expect("SNN creation failed");

        // Run several steps to build up state
        let input = Array1::from_vec(vec![1.0_f32; 4]);
        for _ in 0..10 {
            let _ = model.step(&input).expect("step failed");
        }

        assert!(model.step_count > 0, "step_count should be > 0");

        // Reset should clear state
        model.reset();

        assert_eq!(model.step_count, 0, "step_count should be 0 after reset");
        for state in &model.layer_states {
            assert!(
                state.voltages.iter().all(|&v| v == 0.0),
                "voltages should be zero after reset"
            );
        }
    }

    // 9. test_refractory_clamps_at_reset_value_soft
    //
    // Verifies that during absolute refractory the membrane potential is held at
    // the reset value established when the spike occurred, not allowed to evolve.
    // SoftReset: V_reset = V_new - threshold.
    //
    // We force a spike by pre-loading the voltage well above threshold via the
    // public `voltages` field, then stepping with zero input.  Because
    // new_voltages = state.voltages * leak_factor + W@0 + bias, even with
    // small weights / bias the pre-loaded value ensures new_voltages > threshold.
    #[test]
    fn test_refractory_clamps_at_reset_value_soft() {
        let threshold = 0.5_f32;
        let leak_factor = 0.9_f32;
        let config = SpikingConfig {
            threshold,
            leak_factor,
            refractory_period: 3,
            reset_mode: ResetMode::SoftReset,
            ..SpikingConfig::new(1, 1, 1, 1)
        };
        let layer = LifLayer::new(1, 1, &config).expect("LIF layer creation failed");
        let mut state = layer.init_state();

        // Pre-load voltage far above threshold; with zero input the LIF equation gives
        // new_v = 1000.0 * 0.9 + bias ≈ 900, which is >> threshold → guaranteed spike.
        state.voltages[0] = 1_000.0;
        let zero_input = Array1::from_vec(vec![0.0_f32]);
        let spike_step = layer
            .step(&zero_input, &mut state)
            .expect("spike step failed");

        assert_eq!(
            spike_step[0], 1.0,
            "neuron must spike when voltage is pre-loaded above threshold"
        );

        // state.voltages[0] now holds V_reset = V_new - threshold (SoftReset).
        let reset_value = state.voltages[0];

        // First refractory step: even with nonzero input the voltage must stay at reset_value.
        let nonzero_input = Array1::from_vec(vec![50.0_f32]);
        let refractory_spikes = layer
            .step(&nonzero_input, &mut state)
            .expect("refractory step 1 failed");

        assert_eq!(
            refractory_spikes[0], 0.0,
            "neuron must not spike during refractory"
        );
        assert_eq!(
            state.voltages[0], reset_value,
            "voltage must be clamped at reset value during refractory (SoftReset): \
             expected {reset_value}, got {}",
            state.voltages[0]
        );
    }

    // 10. test_refractory_hard_reset_stays_zero
    //
    // Verifies that HardReset holds the membrane at 0.0 throughout the entire
    // refractory window, even when large synaptic currents arrive.
    #[test]
    fn test_refractory_hard_reset_stays_zero() {
        let threshold = 0.5_f32;
        let config = SpikingConfig {
            threshold,
            leak_factor: 0.9,
            refractory_period: 3,
            reset_mode: ResetMode::HardReset,
            ..SpikingConfig::new(1, 1, 1, 1)
        };
        let layer = LifLayer::new(1, 1, &config).expect("LIF layer creation failed");
        let mut state = layer.init_state();

        // Pre-load voltage above threshold to guarantee a spike with zero input.
        state.voltages[0] = 1_000.0;
        let zero_input = Array1::from_vec(vec![0.0_f32]);
        let spike_step = layer
            .step(&zero_input, &mut state)
            .expect("spike step failed");
        assert_eq!(
            spike_step[0], 1.0,
            "neuron must spike when voltage is pre-loaded above threshold"
        );

        // HardReset sets V to 0.0 at spike time.
        assert_eq!(
            state.voltages[0], 0.0,
            "HardReset must set voltage to 0 at spike time"
        );

        // Run all 3 refractory steps; voltage must remain 0.0 regardless of input.
        let nonzero_input = Array1::from_vec(vec![50.0_f32]);
        for refr_step in 1..=3_u32 {
            let spikes = layer
                .step(&nonzero_input, &mut state)
                .expect("refractory step failed");
            assert_eq!(
                spikes[0], 0.0,
                "no spike allowed during refractory (step {refr_step})"
            );
            assert_eq!(
                state.voltages[0], 0.0,
                "HardReset voltage must stay 0 during refractory (step {refr_step}): got {}",
                state.voltages[0]
            );
        }
    }

    // 11. test_refractory_no_double_leak
    //
    // Verifies that the old bug (applying `state.voltages[i] * leak_factor` during
    // refractory instead of holding the reset value) is fixed.
    //
    // Before the fix, in the first refractory step the code computed:
    //   updated_voltages[i] = state.voltages[i] * leak_factor   (= V_reset * leak_factor)
    // With the fix the voltage must equal V_reset exactly (no leak applied).
    #[test]
    fn test_refractory_no_double_leak() {
        let leak_factor = 0.9_f32;
        let threshold = 0.5_f32;
        let config = SpikingConfig {
            threshold,
            leak_factor,
            refractory_period: 3,
            reset_mode: ResetMode::SoftReset,
            ..SpikingConfig::new(1, 1, 1, 1)
        };
        let layer = LifLayer::new(1, 1, &config).expect("LIF layer creation failed");
        let mut state = layer.init_state();

        // Pre-load voltage to a known value well above threshold; with zero input:
        // new_v = 10.0 * leak_factor + bias ≈ 9.0 (>> threshold).
        // V_reset (SoftReset) = new_v - threshold ≈ 9.0 - 0.5 = 8.5.
        state.voltages[0] = 10.0;
        let zero_input = Array1::from_vec(vec![0.0_f32]);
        let spike_step = layer
            .step(&zero_input, &mut state)
            .expect("spike step failed");
        assert_eq!(spike_step[0], 1.0, "neuron must spike");

        // V_reset stored in state after spike step.
        let v_reset = state.voltages[0];
        assert!(
            v_reset > 0.0,
            "SoftReset value must be positive for this test to be meaningful; got {v_reset}"
        );

        // The old bug produces v_reset * leak_factor after one refractory step.
        // The correct behaviour holds v_reset unchanged.
        let bugged_value = v_reset * leak_factor;

        // Run one refractory step with zero input (isolates just the clamping logic).
        layer
            .step(&zero_input, &mut state)
            .expect("first refractory step failed");

        let v_after = state.voltages[0];

        // Must equal V_reset, not V_reset * leak_factor.
        assert!(
            (v_after - v_reset).abs() < 1e-5,
            "voltage after first refractory step must equal V_reset={v_reset}, \
             got {v_after} (bugged value would be {bugged_value})"
        );
    }

    // 8. test_stdp_update
    #[test]
    fn test_stdp_update() {
        let stdp_config = StdpConfig::default();
        let updater = StdpUpdater::new(stdp_config);

        let mut weights = Array2::zeros((4, 4));
        let pre_traces = Array1::from_vec(vec![1.0_f32; 4]);
        let post_spikes = Array1::from_vec(vec![1.0_f32, 0.0, 1.0, 0.0]);
        let post_traces = Array1::from_vec(vec![0.5_f32; 4]);

        let result = updater.update_weights(&mut weights, &pre_traces, &post_spikes, &post_traces);
        assert!(result.is_ok(), "STDP update failed: {:?}", result);

        // For neurons that spiked (post_spikes[0]=1, post_spikes[2]=1), weights should increase
        // Potentiation: w[0,j] += A+ * pre_traces[j] = 0.01 * 1.0 = 0.01
        // Depression:   w[0,j] -= A- * post_traces[0] * pre_traces[j] = 0.012 * 0.5 * 1.0 = 0.006
        // Net: +0.01 - 0.006 = +0.004 for spiking neurons
        let w00 = weights[[0, 0]];
        assert!(
            w00 > -1e-6,
            "weight for spiking post-neuron should not strongly decrease: got {w00}"
        );
    }
}
