//! Model Interpretability Tools
//!
//! This module provides tools for understanding what RWKV and other SSM models
//! have learned: activation statistics, gating pattern analysis, state trajectory
//! inspection, sensitivity analysis, and compression potential estimation.
//!
//! # Design
//!
//! All analysis is performed in pure Rust using `scirs2-core` arrays.
//! No external Python or visualization dependencies are required.

use crate::error::{ModelError, ModelResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// ActivationStats
// ---------------------------------------------------------------------------

/// Running statistics over a sequence of activations.
///
/// Uses Welford's online algorithm for numerically stable mean/variance.
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// Per-dimension mean
    pub mean: Array1<f32>,
    /// Per-dimension variance (population variance)
    pub variance: Array1<f32>,
    /// Per-dimension maximum encountered value
    pub max: Array1<f32>,
    /// Per-dimension minimum encountered value
    pub min: Array1<f32>,
    /// Fraction of values with absolute value below `eps` (default 1e-6)
    pub sparsity: f32,
    /// Global L2 norm (average over all steps)
    pub l2_norm: f32,
    /// Number of activation vectors accumulated
    pub num_steps: usize,

    // Welford running state (not serialized / public)
    welford_m2: Array1<f32>,
    near_zero_count: usize,
    total_elements: usize,
    l2_sum: f32,
}

impl ActivationStats {
    /// Compute statistics from a fixed batch of activations.
    ///
    /// Returns an error if the slice is empty or dimensions are inconsistent.
    pub fn from_sequence(activations: &[Array1<f32>]) -> ModelResult<Self> {
        if activations.is_empty() {
            return Err(ModelError::invalid_config(
                "ActivationStats::from_sequence: empty activation sequence",
            ));
        }
        let dim = activations[0].len();
        for (i, a) in activations.iter().enumerate() {
            if a.len() != dim {
                return Err(ModelError::dimension_mismatch(
                    format!("activation[{i}]"),
                    dim,
                    a.len(),
                ));
            }
        }

        let mut stats = Self::zero(dim);
        for a in activations {
            stats.update(a);
        }
        Ok(stats)
    }

    /// Create a zeroed stats object for the given dimension.
    fn zero(dim: usize) -> Self {
        Self {
            mean: Array1::zeros(dim),
            variance: Array1::zeros(dim),
            max: Array1::from_elem(dim, f32::NEG_INFINITY),
            min: Array1::from_elem(dim, f32::INFINITY),
            sparsity: 0.0,
            l2_norm: 0.0,
            num_steps: 0,
            welford_m2: Array1::zeros(dim),
            near_zero_count: 0,
            total_elements: 0,
            l2_sum: 0.0,
        }
    }

    /// Incrementally incorporate a new activation vector (Welford's algorithm).
    pub fn update(&mut self, activation: &Array1<f32>) {
        let eps = 1e-6_f32;
        self.num_steps += 1;
        let n = self.num_steps as f32;

        let mut sq_sum = 0.0_f32;
        let mut nz = 0usize;

        for (i, &v) in activation.iter().enumerate() {
            if i >= self.mean.len() {
                break;
            }
            // Welford update
            let delta = v - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = v - self.mean[i];
            self.welford_m2[i] += delta * delta2;
            self.variance[i] = if self.num_steps > 1 {
                self.welford_m2[i] / n
            } else {
                0.0
            };

            // Min / max
            if v > self.max[i] {
                self.max[i] = v;
            }
            if v < self.min[i] {
                self.min[i] = v;
            }

            // Near-zero count
            if v.abs() < eps {
                nz += 1;
            }

            sq_sum += v * v;
        }

        self.near_zero_count += nz;
        self.total_elements += activation.len();
        self.l2_sum += sq_sum.sqrt();
        self.l2_norm = self.l2_sum / n;
        self.sparsity = if self.total_elements > 0 {
            self.near_zero_count as f32 / self.total_elements as f32
        } else {
            0.0
        };
    }

    /// Reset all statistics back to zero.
    pub fn reset(&mut self) {
        let dim = self.mean.len();
        self.mean.fill(0.0);
        self.variance.fill(0.0);
        self.max.fill(f32::NEG_INFINITY);
        self.min.fill(f32::INFINITY);
        self.sparsity = 0.0;
        self.l2_norm = 0.0;
        self.num_steps = 0;
        self.welford_m2 = Array1::zeros(dim);
        self.near_zero_count = 0;
        self.total_elements = 0;
        self.l2_sum = 0.0;
    }
}

// ---------------------------------------------------------------------------
// LayerProbe
// ---------------------------------------------------------------------------

/// Ring-buffer probe for capturing intermediate layer outputs.
///
/// When `enabled`, each call to [`LayerProbe::capture`] stores the activation.
/// The buffer is bounded to `max_capture` entries; older entries are overwritten.
pub struct LayerProbe {
    layer_name: String,
    captured: Vec<Array1<f32>>,
    max_capture: usize,
    head: usize, // write index into ring buffer (wraps around)
    filled: bool,
    enabled: bool,
}

impl LayerProbe {
    /// Create a new probe for `layer_name` that holds up to `max_capture` activations.
    pub fn new(layer_name: &str, max_capture: usize) -> Self {
        let max_capture = max_capture.max(1);
        Self {
            layer_name: layer_name.to_owned(),
            captured: Vec::with_capacity(max_capture),
            max_capture,
            head: 0,
            filled: false,
            enabled: true,
        }
    }

    /// Capture one activation vector. No-op if disabled.
    pub fn capture(&mut self, activation: Array1<f32>) {
        if !self.enabled {
            return;
        }
        if self.captured.len() < self.max_capture {
            self.captured.push(activation);
        } else {
            self.captured[self.head] = activation;
            self.filled = true;
        }
        self.head = (self.head + 1) % self.max_capture;
    }

    /// Compute statistics over all captured activations.
    pub fn stats(&self) -> ModelResult<ActivationStats> {
        if self.captured.is_empty() {
            return Err(ModelError::invalid_config(format!(
                "LayerProbe '{}': no activations captured",
                self.layer_name
            )));
        }
        ActivationStats::from_sequence(&self.captured)
    }

    /// Read all captured activations (in capture order if not wrapped, otherwise ring order).
    pub fn activations(&self) -> &[Array1<f32>] {
        &self.captured
    }

    /// Whether the ring buffer has wrapped around at least once.
    pub fn is_full(&self) -> bool {
        self.filled
    }

    /// Enable capturing.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable capturing (future captures are silently dropped).
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Clear all captured activations and reset ring-buffer state.
    pub fn clear(&mut self) {
        self.captured.clear();
        self.head = 0;
        self.filled = false;
    }

    /// Return the layer name associated with this probe.
    pub fn layer_name(&self) -> &str {
        &self.layer_name
    }
}

// ---------------------------------------------------------------------------
// GatingAnalysis
// ---------------------------------------------------------------------------

/// Analysis of gating patterns in gated models (RWKV, Mamba, etc.).
#[derive(Debug, Clone)]
pub struct GatingAnalysis {
    /// All captured gate activation vectors (one per step)
    pub gate_values: Vec<Array1<f32>>,
    /// Average gate value per unit, across all steps
    pub avg_gate: Array1<f32>,
    /// Indices of "dead" gates: average value below `threshold`
    pub dead_gates: Vec<usize>,
    /// Indices of "saturated" gates: average value above `1 - threshold`
    pub saturated_gates: Vec<usize>,
    /// Shannon entropy of the average gate distribution (treating avg_gate as unnormalised probs)
    pub gate_entropy: f32,
}

impl GatingAnalysis {
    /// Analyse the given gate activations.
    ///
    /// # Parameters
    ///
    /// - `gate_values` — one `Array1<f32>` per time-step; all must have the same length
    /// - `threshold` — gates with avg < threshold are "dead"; avg > 1 - threshold are "saturated"
    pub fn from_activations(gate_values: Vec<Array1<f32>>, threshold: f32) -> ModelResult<Self> {
        if gate_values.is_empty() {
            return Err(ModelError::invalid_config(
                "GatingAnalysis: no gate values provided",
            ));
        }
        let dim = gate_values[0].len();
        for (i, g) in gate_values.iter().enumerate() {
            if g.len() != dim {
                return Err(ModelError::dimension_mismatch(
                    format!("gate_values[{i}]"),
                    dim,
                    g.len(),
                ));
            }
        }

        // Compute average gate per dimension
        let n = gate_values.len() as f32;
        let mut avg_gate = Array1::zeros(dim);
        for g in &gate_values {
            for (i, &v) in g.iter().enumerate() {
                avg_gate[i] += v;
            }
        }
        avg_gate.mapv_inplace(|v: f32| v / n);

        let threshold_clamped = threshold.clamp(0.0, 0.5);
        let mut dead_gates = Vec::new();
        let mut saturated_gates = Vec::new();
        for (i, &v) in avg_gate.iter().enumerate() {
            if v < threshold_clamped {
                dead_gates.push(i);
            } else if v > 1.0 - threshold_clamped {
                saturated_gates.push(i);
            }
        }

        // Shannon entropy: H = -sum p*log2(p), treating each avg as a probability of being open
        let eps = 1e-9_f32;
        let mut entropy = 0.0_f32;
        for &p in avg_gate.iter() {
            let p: f32 = p.clamp(eps, 1.0 - eps);
            entropy -= p * p.log2() + (1.0 - p) * (1.0 - p).log2();
        }
        let gate_entropy = entropy / dim as f32;

        Ok(Self {
            gate_values,
            avg_gate,
            dead_gates,
            saturated_gates,
            gate_entropy,
        })
    }

    /// Fraction of gates that are neither dead nor saturated (effective gates).
    pub fn effective_capacity(&self) -> f32 {
        let total = self.avg_gate.len();
        if total == 0 {
            return 0.0;
        }
        let inactive = self.dead_gates.len() + self.saturated_gates.len();
        let active = total.saturating_sub(inactive);
        active as f32 / total as f32
    }
}

// ---------------------------------------------------------------------------
// StateTrajectory
// ---------------------------------------------------------------------------

/// Records and analyses the trajectory of hidden states over time.
pub struct StateTrajectory {
    states: Vec<Array1<f32>>,
    dim: usize,
}

impl StateTrajectory {
    /// Create a new trajectory recorder for the given state dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            states: Vec::new(),
            dim,
        }
    }

    /// Append a state vector; returns an error on dimension mismatch.
    pub fn push(&mut self, state: Array1<f32>) -> ModelResult<()> {
        if state.len() != self.dim {
            return Err(ModelError::dimension_mismatch(
                "StateTrajectory::push",
                self.dim,
                state.len(),
            ));
        }
        self.states.push(state);
        Ok(())
    }

    /// Number of states recorded.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether no states have been recorded.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Compute per-step velocities: `||s_{t+1} - s_t||_2` for `t` in `0..len-1`.
    ///
    /// Returns an error if fewer than 2 states have been recorded.
    pub fn velocities(&self) -> ModelResult<Vec<f32>> {
        if self.states.len() < 2 {
            return Err(ModelError::invalid_config(
                "StateTrajectory::velocities: need at least 2 states",
            ));
        }
        let mut vels = Vec::with_capacity(self.states.len() - 1);
        for w in self.states.windows(2) {
            let diff = &w[1] - &w[0];
            let norm = diff.iter().map(|&v| v * v).sum::<f32>().sqrt();
            vels.push(norm);
        }
        Ok(vels)
    }

    /// Effective dimensionality via participation ratio.
    ///
    /// PR = (Σ λ_i)² / Σ λ_i²  where λ_i are per-dimension variances.
    /// A high value means the state uses many dimensions; a low value means
    /// information is concentrated in a few dimensions.
    pub fn participation_ratio(&self) -> ModelResult<f32> {
        if self.states.is_empty() {
            return Err(ModelError::invalid_config(
                "StateTrajectory::participation_ratio: no states recorded",
            ));
        }

        // Compute per-dimension variance
        let n = self.states.len() as f32;
        let mut mean: Array1<f32> = Array1::zeros(self.dim);
        for s in &self.states {
            for (i, &v) in s.iter().enumerate() {
                mean[i] += v;
            }
        }
        mean.mapv_inplace(|v: f32| v / n);

        let mut var: Array1<f32> = Array1::zeros(self.dim);
        for s in &self.states {
            for (i, &v) in s.iter().enumerate() {
                let d: f32 = v - mean[i];
                var[i] += d * d;
            }
        }
        var.mapv_inplace(|v: f32| v / n);

        let sum_var: f32 = var.iter().sum();
        let sum_var_sq: f32 = var.iter().map(|&v| v * v).sum();

        if sum_var_sq < 1e-20 {
            // All states identical → effectively 0-dimensional
            return Ok(0.0);
        }

        Ok((sum_var * sum_var) / sum_var_sq)
    }

    /// Return indices of the `k` dimensions with highest variance, sorted descending.
    pub fn most_variable_dims(&self, k: usize) -> ModelResult<Vec<usize>> {
        if self.states.is_empty() {
            return Err(ModelError::invalid_config(
                "StateTrajectory::most_variable_dims: no states recorded",
            ));
        }
        let k = k.min(self.dim);

        let n = self.states.len() as f32;
        let mut mean = vec![0.0_f32; self.dim];
        for s in &self.states {
            for (i, &v) in s.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in &mut mean {
            *m /= n;
        }

        let mut var = vec![0.0_f32; self.dim];
        for s in &self.states {
            for (i, &v) in s.iter().enumerate() {
                let d = v - mean[i];
                var[i] += d * d;
            }
        }
        for v in &mut var {
            *v /= n;
        }

        let mut idx: Vec<usize> = (0..self.dim).collect();
        idx.sort_unstable_by(|&a, &b| {
            var[b]
                .partial_cmp(&var[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(k);
        Ok(idx)
    }

    /// Compute state autocorrelation at lag `lag`.
    ///
    /// Returns the mean Pearson correlation of `s_t` and `s_{t+lag}` across all
    /// valid pairs.  A lag-0 autocorrelation is always 1.0.
    pub fn autocorrelation(&self, lag: usize) -> ModelResult<f32> {
        if self.states.len() <= lag {
            return Err(ModelError::invalid_config(format!(
                "StateTrajectory::autocorrelation: lag {lag} requires at least {} states, have {}",
                lag + 1,
                self.states.len()
            )));
        }

        let n_pairs = self.states.len() - lag;
        let mut corr_sum = 0.0_f32;

        for t in 0..n_pairs {
            let s0 = &self.states[t];
            let s1 = &self.states[t + lag];

            // Pearson correlation between s0 and s1 (as two length-dim vectors)
            let n = self.dim as f32;
            let mean0: f32 = s0.iter().sum::<f32>() / n;
            let mean1: f32 = s1.iter().sum::<f32>() / n;

            let mut cov = 0.0_f32;
            let mut std0 = 0.0_f32;
            let mut std1 = 0.0_f32;
            for (&a, &b) in s0.iter().zip(s1.iter()) {
                let da = a - mean0;
                let db = b - mean1;
                cov += da * db;
                std0 += da * da;
                std1 += db * db;
            }

            let denom = (std0 * std1).sqrt();
            if denom < 1e-10 {
                // Constant vectors → perfectly correlated by convention
                corr_sum += 1.0;
            } else {
                corr_sum += cov / denom;
            }
        }

        Ok(corr_sum / n_pairs as f32)
    }

    /// Flatten all states into an `(num_steps, dim)` matrix.
    pub fn to_matrix(&self) -> ModelResult<Array2<f32>> {
        if self.states.is_empty() {
            return Err(ModelError::invalid_config(
                "StateTrajectory::to_matrix: no states recorded",
            ));
        }
        let t = self.states.len();
        let d = self.dim;
        let mut mat = Array2::zeros((t, d));
        for (row, state) in self.states.iter().enumerate() {
            for (col, &v) in state.iter().enumerate() {
                mat[[row, col]] = v;
            }
        }
        Ok(mat)
    }
}

// ---------------------------------------------------------------------------
// SensitivityAnalyzer
// ---------------------------------------------------------------------------

/// Measures feature importance via finite-difference sensitivity analysis.
pub struct SensitivityAnalyzer {
    input_dim: usize,
}

impl SensitivityAnalyzer {
    /// Create an analyzer for inputs of size `input_dim`.
    pub fn new(input_dim: usize) -> Self {
        Self { input_dim }
    }

    /// Estimate per-feature sensitivity using finite differences.
    ///
    /// For each feature `i`, computes `||f(x + eps*e_i) - f(x)|| / eps`.
    ///
    /// # Arguments
    ///
    /// - `input` — base input vector (length must equal `input_dim`)
    /// - `forward_fn` — model forward pass (called `input_dim + 1` times)
    /// - `eps` — perturbation magnitude (default 1e-3 is reasonable)
    pub fn input_sensitivity<F>(
        &self,
        input: &Array1<f32>,
        forward_fn: F,
        eps: f32,
    ) -> ModelResult<Array1<f32>>
    where
        F: Fn(&Array1<f32>) -> ModelResult<Array1<f32>>,
    {
        if input.len() != self.input_dim {
            return Err(ModelError::dimension_mismatch(
                "SensitivityAnalyzer::input_sensitivity",
                self.input_dim,
                input.len(),
            ));
        }

        let base_out = forward_fn(input)?;
        let base_norm = base_out.iter().map(|&v| v * v).sum::<f32>().sqrt();

        let mut sensitivities = Array1::zeros(self.input_dim);
        for i in 0..self.input_dim {
            let mut perturbed = input.clone();
            perturbed[i] += eps;
            let pert_out = forward_fn(&perturbed)?;

            // Measure change in output norm
            let diff_norm = pert_out
                .iter()
                .zip(base_out.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();

            sensitivities[i] = if eps.abs() > 1e-15 {
                diff_norm / eps.abs()
            } else {
                base_norm
            };
        }

        Ok(sensitivities)
    }

    /// Rank features by average sensitivity across multiple input vectors.
    ///
    /// Returns a sorted list of `(feature_index, avg_sensitivity)` in descending order.
    pub fn rank_features<F>(
        &self,
        inputs: &[Array1<f32>],
        forward_fn: F,
        eps: f32,
    ) -> ModelResult<Vec<(usize, f32)>>
    where
        F: Fn(&Array1<f32>) -> ModelResult<Array1<f32>>,
    {
        if inputs.is_empty() {
            return Err(ModelError::invalid_config(
                "SensitivityAnalyzer::rank_features: no inputs provided",
            ));
        }

        let mut total: Array1<f32> = Array1::zeros(self.input_dim);
        for input in inputs {
            let sens = self.input_sensitivity(input, &forward_fn, eps)?;
            for (i, &v) in sens.iter().enumerate() {
                total[i] += v;
            }
        }

        let n = inputs.len() as f32;
        let mut ranked: Vec<(usize, f32)> =
            total.iter().enumerate().map(|(i, &v)| (i, v / n)).collect();

        ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(ranked)
    }
}

// ---------------------------------------------------------------------------
// CompressionAnalysis
// ---------------------------------------------------------------------------

/// Estimates how compressible a weight matrix is.
#[derive(Debug, Clone)]
pub struct CompressionAnalysis {
    /// Fraction of weight values with |w| < eps
    pub weight_sparsity: f32,
    /// Effective rank via participation ratio (higher = less compressible)
    pub effective_rank: f32,
    /// Estimated INT8 quantization relative error
    pub quantization_error: f32,
    /// Suggested LoRA rank for this layer (heuristic)
    pub recommended_rank: usize,
    /// Overall compressibility score in [0, 1] (1 = very compressible)
    pub compression_potential: f32,
}

impl CompressionAnalysis {
    /// Analyse a single weight matrix.
    ///
    /// `eps` is the threshold below which a weight is considered "zero".
    pub fn analyze_weight(weight: &Array2<f32>, eps: f32) -> ModelResult<Self> {
        let (rows, cols) = (weight.shape()[0], weight.shape()[1]);
        let total = rows * cols;

        if total == 0 {
            return Err(ModelError::invalid_config(
                "CompressionAnalysis: weight matrix is empty",
            ));
        }

        // --- Sparsity ---
        let near_zero = weight.iter().filter(|&&v| v.abs() < eps).count();
        let weight_sparsity = near_zero as f32 / total as f32;

        // --- Per-column variances as surrogate for singular values ---
        // (Full SVD would be expensive; variance-based PR is a good approximation)
        let n = rows as f32;
        let col_variances: Vec<f32> = (0..cols)
            .map(|j| {
                let col = weight.column(j);
                let mean = col.iter().sum::<f32>() / n;
                col.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n
            })
            .collect();

        let sum_var: f32 = col_variances.iter().sum();
        let sum_var_sq: f32 = col_variances.iter().map(|&v| v * v).sum();
        let effective_rank = if sum_var_sq > 1e-20 {
            (sum_var * sum_var) / sum_var_sq
        } else {
            1.0
        };

        // --- INT8 quantization error estimate ---
        // max_abs * scale_error, where scale_error ≈ 1/(2^8 - 1)
        let max_abs = weight.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let quantization_error = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };

        // --- Recommended LoRA rank (heuristic: ceil(effective_rank / 4)) ---
        let recommended_rank = ((effective_rank / 4.0).ceil() as usize).max(1);

        // --- Compression potential ---
        // Combines sparsity and low effective rank
        let max_dim = rows.max(cols) as f32;
        let rank_score = 1.0 - (effective_rank / max_dim).clamp(0.0, 1.0);
        let compression_potential = (0.6 * rank_score + 0.4 * weight_sparsity).clamp(0.0, 1.0);

        Ok(Self {
            weight_sparsity,
            effective_rank,
            quantization_error,
            recommended_rank,
            compression_potential,
        })
    }

    /// Analyse multiple weight matrices, returning a list of (name, analysis) pairs.
    pub fn analyze_multiple<'a>(
        weights: &[(&'a str, &Array2<f32>)],
    ) -> Vec<(&'a str, CompressionAnalysis)> {
        weights
            .iter()
            .filter_map(|&(name, w)| Self::analyze_weight(w, 1e-6).ok().map(|a| (name, a)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// InterpretabilityReport
// ---------------------------------------------------------------------------

/// High-level interpretability summary collected from a single model run.
pub struct InterpretabilityReport {
    /// Total number of steps included in this report
    pub num_steps: usize,
    /// Per-layer activation statistics: (layer_name, stats)
    pub layer_stats: Vec<(String, ActivationStats)>,
    /// State trajectory of the primary hidden state
    pub state_trajectory: StateTrajectory,
    /// Feature sensitivity ranking (feature_index, avg_sensitivity)
    pub top_sensitive_features: Vec<(usize, f32)>,
    /// Overall fraction of near-zero activations across all layers
    pub overall_sparsity: f32,
}

impl Default for InterpretabilityReport {
    fn default() -> Self {
        Self::new()
    }
}

impl InterpretabilityReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self {
            num_steps: 0,
            layer_stats: Vec::new(),
            state_trajectory: StateTrajectory::new(0),
            top_sensitive_features: Vec::new(),
            overall_sparsity: 0.0,
        }
    }

    /// Generate a human-readable summary string.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "InterpretabilityReport — {} step(s), {} layer(s)",
            self.num_steps,
            self.layer_stats.len()
        ));
        lines.push(format!(
            "  Overall sparsity : {:.2}%",
            self.overall_sparsity * 100.0
        ));
        lines.push(format!(
            "  State trajectory : {} entries, dim={}",
            self.state_trajectory.len(),
            self.state_trajectory.dim
        ));

        if !self.layer_stats.is_empty() {
            lines.push("  Layer statistics:".to_owned());
            for (name, stats) in &self.layer_stats {
                lines.push(format!(
                    "    {name}: sparsity={:.2}% l2={:.4} steps={}",
                    stats.sparsity * 100.0,
                    stats.l2_norm,
                    stats.num_steps
                ));
            }
        }

        if !self.top_sensitive_features.is_empty() {
            lines.push("  Top sensitive features:".to_owned());
            for &(idx, sens) in self.top_sensitive_features.iter().take(5) {
                lines.push(format!("    feature {idx}: {sens:.4}"));
            }
        }

        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // -----------------------------------------------------------------------
    // Test 1: ActivationStats basic
    // -----------------------------------------------------------------------
    #[test]
    fn test_activation_stats_basic() {
        // Five identical vectors: mean == value, variance == 0
        let v = array![1.0_f32, 2.0, 3.0];
        let activations: Vec<Array1<f32>> = (0..5).map(|_| v.clone()).collect();
        let stats = ActivationStats::from_sequence(&activations).expect("stats");

        assert_eq!(stats.num_steps, 5);
        for (&m, &expected) in stats.mean.iter().zip(v.iter()) {
            assert!((m - expected).abs() < 1e-5, "mean mismatch");
        }
        for &var in stats.variance.iter() {
            assert!(
                var.abs() < 1e-5,
                "variance should be ~0 for identical vectors"
            );
        }
        // No near-zero elements (all >= 1.0)
        assert_eq!(stats.sparsity, 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 2: Incremental vs batch
    // -----------------------------------------------------------------------
    #[test]
    fn test_activation_stats_incremental() {
        let activations: Vec<Array1<f32>> =
            (0..10).map(|i| array![i as f32, (i * 2) as f32]).collect();

        let batch = ActivationStats::from_sequence(&activations).expect("batch");

        let mut incr = ActivationStats::zero(2);
        for a in &activations {
            incr.update(a);
        }

        for (&bm, &im) in batch.mean.iter().zip(incr.mean.iter()) {
            assert!((bm - im).abs() < 1e-4, "mean mismatch: {bm} vs {im}");
        }
        for (&bv, &iv) in batch.variance.iter().zip(incr.variance.iter()) {
            assert!((bv - iv).abs() < 1e-4, "variance mismatch: {bv} vs {iv}");
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: LayerProbe capture count
    // -----------------------------------------------------------------------
    #[test]
    fn test_layer_probe_capture() {
        let mut probe = LayerProbe::new("layer0", 1000);
        assert!(probe.activations().is_empty());

        for i in 0..7 {
            probe.capture(array![i as f32, 0.0]);
        }
        assert_eq!(probe.activations().len(), 7);
        assert_eq!(probe.layer_name(), "layer0");

        // Disable → capture is ignored
        probe.disable();
        probe.capture(array![99.0, 0.0]);
        assert_eq!(probe.activations().len(), 7);

        // Clear
        probe.enable();
        probe.clear();
        assert!(probe.activations().is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 4: LayerProbe stats
    // -----------------------------------------------------------------------
    #[test]
    fn test_layer_probe_stats() {
        let mut probe = LayerProbe::new("attn", 100);
        probe.capture(array![0.0_f32, 0.0]);
        probe.capture(array![2.0_f32, 4.0]);

        let stats = probe.stats().expect("stats");
        assert_eq!(stats.num_steps, 2);
        // Mean should be [1.0, 2.0]
        assert!((stats.mean[0] - 1.0).abs() < 1e-5);
        assert!((stats.mean[1] - 2.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Test 5: StateTrajectory velocities shape
    // -----------------------------------------------------------------------
    #[test]
    fn test_state_trajectory_velocities() {
        let mut traj = StateTrajectory::new(4);
        for i in 0..5_u32 {
            traj.push(Array1::from_elem(4, i as f32)).expect("push");
        }
        let vels = traj.velocities().expect("velocities");
        assert_eq!(vels.len(), 4, "should have len-1 velocities");
        for v in &vels {
            assert!(v.is_finite(), "velocities must be finite");
            assert!(*v >= 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // Test 6: Lag-0 autocorrelation == 1.0
    // -----------------------------------------------------------------------
    #[test]
    fn test_state_trajectory_autocorrelation() {
        let mut traj = StateTrajectory::new(8);
        for i in 0..10_u32 {
            let s = Array1::from_shape_fn(8, |j| (i * 8 + j as u32) as f32);
            traj.push(s).expect("push");
        }

        let ac0 = traj.autocorrelation(0).expect("lag0");
        assert!(
            (ac0 - 1.0).abs() < 1e-5,
            "lag-0 autocorr should be 1.0, got {ac0}"
        );

        // Lag-1 should be finite and in [-1, 1]
        let ac1 = traj.autocorrelation(1).expect("lag1");
        assert!(ac1.is_finite());
        assert!((-1.0_f32..=1.0_f32).contains(&ac1));
    }

    // -----------------------------------------------------------------------
    // Test 7: SensitivityAnalyzer gives non-negative sensitivities
    // -----------------------------------------------------------------------
    #[test]
    fn test_sensitivity_analyzer() {
        let analyzer = SensitivityAnalyzer::new(3);

        // Forward function: identity (output == input)
        let forward = |x: &Array1<f32>| -> ModelResult<Array1<f32>> { Ok(x.clone()) };

        let input = array![1.0_f32, -0.5, 2.0];
        let sens = analyzer
            .input_sensitivity(&input, forward, 1e-3)
            .expect("sensitivity");

        assert_eq!(sens.len(), 3);
        for &s in sens.iter() {
            assert!(s >= 0.0, "sensitivity must be non-negative, got {s}");
            assert!(s.is_finite());
        }
    }

    // -----------------------------------------------------------------------
    // Test 8: CompressionAnalysis valid ratios
    // -----------------------------------------------------------------------
    #[test]
    fn test_compression_analysis() {
        let w: Array2<f32> =
            Array2::from_shape_fn((16, 16), |(i, j)| if i == j { 1.0 } else { 0.0 });
        let analysis = CompressionAnalysis::analyze_weight(&w, 1e-6).expect("analysis");

        // Identity matrix: nearly all zeros → high sparsity
        assert!((0.0..=1.0_f32).contains(&analysis.weight_sparsity));
        assert!(analysis.compression_potential >= 0.0);
        assert!(analysis.compression_potential <= 1.0);
        assert!(analysis.effective_rank > 0.0);
        assert!(analysis.recommended_rank >= 1);
    }
}
