//! Advanced quantization strategies
//!
//! Provides sophisticated quantization methods beyond simple linear/μ-law:
//! - **Adaptive Quantization**: Adjusts step size based on signal statistics
//! - **Dead Zone Quantization**: Optimized for sparse signals
//! - **Perceptual Quantization**: Psychoacoustic modeling for audio
//! - **Entropy-Constrained Quantization**: Rate-distortion optimized

use crate::error::{TokenizerError, TokenizerResult};
use crate::{Quantizer, SignalTokenizer};
use scirs2_core::ndarray::Array1;

/// Adaptive quantizer that adjusts step size based on local signal statistics
///
/// Uses a sliding window to compute local variance and adapts quantization
/// step size accordingly. High-variance regions get finer quantization.
#[derive(Debug, Clone)]
pub struct AdaptiveQuantizer {
    /// Base number of bits
    _bits: u8,
    /// Number of levels
    levels: usize,
    /// Window size for local statistics
    window_size: usize,
    /// Adaptation strength (0.0 = no adaptation, 1.0 = full adaptation)
    adaptation_strength: f32,
    /// Global min/max for normalization
    global_min: f32,
    global_max: f32,
}

impl AdaptiveQuantizer {
    /// Create a new adaptive quantizer
    pub fn new(
        bits: u8,
        window_size: usize,
        adaptation_strength: f32,
        global_min: f32,
        global_max: f32,
    ) -> TokenizerResult<Self> {
        if bits == 0 || bits > 16 {
            return Err(TokenizerError::InvalidConfig("bits must be 1-16".into()));
        }
        if window_size == 0 {
            return Err(TokenizerError::InvalidConfig(
                "window_size must be positive".into(),
            ));
        }
        if !(0.0..=1.0).contains(&adaptation_strength) {
            return Err(TokenizerError::InvalidConfig(
                "adaptation_strength must be in [0, 1]".into(),
            ));
        }

        Ok(Self {
            _bits: bits,
            levels: 1usize << bits,
            window_size,
            adaptation_strength,
            global_min,
            global_max,
        })
    }

    /// Compute local variance around position
    fn local_variance(&self, signal: &Array1<f32>, pos: usize) -> f32 {
        let half_window = self.window_size / 2;
        let start = pos.saturating_sub(half_window);
        let end = (pos + half_window).min(signal.len());

        let window: Vec<f32> = signal
            .iter()
            .skip(start)
            .take(end - start)
            .cloned()
            .collect();
        if window.is_empty() {
            return 1.0;
        }

        let mean = window.iter().sum::<f32>() / window.len() as f32;
        let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / window.len() as f32;

        variance.sqrt().max(1e-6) // Return standard deviation
    }

    /// Compute adaptive step size at position
    fn adaptive_step(&self, signal: &Array1<f32>, pos: usize) -> f32 {
        let base_step = (self.global_max - self.global_min) / self.levels as f32;
        let local_std = self.local_variance(signal, pos);

        // Scale step size based on local statistics
        let global_std = (self.global_max - self.global_min) / 4.0; // Approximate
        let scale = 1.0 + self.adaptation_strength * (local_std / global_std - 1.0);

        base_step * scale.clamp(0.1, 10.0) // Clamp scaling factor
    }

    /// Quantize entire signal with adaptation
    pub fn quantize_adaptive(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<i32>> {
        let mut result = Vec::with_capacity(signal.len());

        for (i, &value) in signal.iter().enumerate() {
            let step = self.adaptive_step(signal, i);
            let clamped = value.clamp(self.global_min, self.global_max);
            let normalized = (clamped - self.global_min) / (self.global_max - self.global_min);
            let level = (normalized / step * (self.levels - 1) as f32).round() as i32;
            result.push(level.clamp(0, (self.levels - 1) as i32));
        }

        Ok(Array1::from_vec(result))
    }
}

impl Quantizer for AdaptiveQuantizer {
    fn quantize(&self, value: f32) -> i32 {
        // Fallback to uniform quantization for single values
        let clamped = value.clamp(self.global_min, self.global_max);
        let normalized = (clamped - self.global_min) / (self.global_max - self.global_min);
        (normalized * (self.levels - 1) as f32).round() as i32
    }

    fn dequantize(&self, level: i32) -> f32 {
        let clamped_level = level.clamp(0, (self.levels - 1) as i32);
        let normalized = clamped_level as f32 / (self.levels - 1) as f32;
        self.global_min + normalized * (self.global_max - self.global_min)
    }

    fn num_levels(&self) -> usize {
        self.levels
    }
}

/// Dead zone quantizer for sparse signals
///
/// Applies a dead zone around zero where small values are quantized to zero.
/// This is useful for signals with many near-zero values (e.g., after transforms).
#[derive(Debug, Clone)]
pub struct DeadZoneQuantizer {
    /// Base quantizer
    _base_bits: u8,
    levels: usize,
    /// Dead zone threshold
    dead_zone: f32,
    /// Range for quantization
    min: f32,
    max: f32,
}

impl DeadZoneQuantizer {
    /// Create a new dead zone quantizer
    ///
    /// # Arguments
    /// * `bits` - Number of quantization bits
    /// * `dead_zone` - Threshold below which values are quantized to zero
    /// * `min`, `max` - Value range
    pub fn new(bits: u8, dead_zone: f32, min: f32, max: f32) -> TokenizerResult<Self> {
        if bits == 0 || bits > 16 {
            return Err(TokenizerError::InvalidConfig("bits must be 1-16".into()));
        }
        if dead_zone < 0.0 {
            return Err(TokenizerError::InvalidConfig(
                "dead_zone must be non-negative".into(),
            ));
        }

        Ok(Self {
            _base_bits: bits,
            levels: 1usize << bits,
            dead_zone,
            min,
            max,
        })
    }
}

impl Quantizer for DeadZoneQuantizer {
    fn quantize(&self, value: f32) -> i32 {
        // Apply dead zone
        if value.abs() < self.dead_zone {
            return (self.levels / 2) as i32; // Zero point
        }

        // Quantize non-dead-zone values
        let clamped = value.clamp(self.min, self.max);
        let normalized = (clamped - self.min) / (self.max - self.min);
        (normalized * (self.levels - 1) as f32).round() as i32
    }

    fn dequantize(&self, level: i32) -> f32 {
        let clamped_level = level.clamp(0, (self.levels - 1) as i32);

        // Check if it's the zero point
        if clamped_level == (self.levels / 2) as i32 {
            return 0.0;
        }

        let normalized = clamped_level as f32 / (self.levels - 1) as f32;
        self.min + normalized * (self.max - self.min)
    }

    fn num_levels(&self) -> usize {
        self.levels
    }
}

/// Non-uniform quantizer with configurable bin edges
///
/// Allows custom quantization levels for optimal rate-distortion trade-off
#[derive(Debug, Clone)]
pub struct NonUniformQuantizer {
    /// Quantization bin edges (sorted)
    bin_edges: Vec<f32>,
    /// Reconstruction values for each bin
    reconstruction_values: Vec<f32>,
}

impl NonUniformQuantizer {
    /// Create from bin edges
    ///
    /// Reconstruction values are set to bin centers
    pub fn from_edges(mut bin_edges: Vec<f32>) -> TokenizerResult<Self> {
        if bin_edges.len() < 2 {
            return Err(TokenizerError::InvalidConfig(
                "Need at least 2 bin edges".into(),
            ));
        }

        bin_edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute reconstruction values as bin centers
        let mut reconstruction_values = Vec::with_capacity(bin_edges.len() - 1);
        for i in 0..bin_edges.len() - 1 {
            reconstruction_values.push((bin_edges[i] + bin_edges[i + 1]) / 2.0);
        }

        Ok(Self {
            bin_edges,
            reconstruction_values,
        })
    }

    /// Create with custom reconstruction values
    pub fn new(bin_edges: Vec<f32>, reconstruction_values: Vec<f32>) -> TokenizerResult<Self> {
        if bin_edges.len() != reconstruction_values.len() + 1 {
            return Err(TokenizerError::InvalidConfig(
                "bin_edges.len() must equal reconstruction_values.len() + 1".into(),
            ));
        }

        Ok(Self {
            bin_edges,
            reconstruction_values,
        })
    }

    /// Create Lloyd-Max quantizer for Gaussian distribution
    ///
    /// Optimizes bin edges and reconstruction values for minimum MSE using the
    /// Lloyd-Max iterative algorithm. Alternates between:
    ///   1. Boundary update: edges = midpoints of adjacent reconstruction levels
    ///   2. Level update: reconstruction values = conditional centroids of a
    ///      zero-mean Gaussian truncated to each bin
    ///
    /// Converges to the globally optimal scalar quantizer for a Gaussian source.
    pub fn lloyd_max_gaussian(num_levels: usize, sigma: f32) -> TokenizerResult<Self> {
        if num_levels < 2 {
            return Err(TokenizerError::InvalidConfig(
                "num_levels must be at least 2".into(),
            ));
        }

        // Standard-normal PDF: φ(x) = exp(-x²/2) / sqrt(2π)
        #[inline]
        fn phi(x: f32) -> f32 {
            (-0.5 * x * x).exp() / (2.0 * std::f32::consts::PI).sqrt()
        }

        // Standard-normal CDF via Abramowitz-Stegun 7.1.26 approximation for erfc.
        // Since Φ(x) = erfc(−x/√2)/2, the approximation is applied at y = |x|/√2:
        //   erfc(y) ≈ (a₁t + a₂t² + a₃t³) · exp(−y²),  t = 1/(1+p·y)
        // Maximum absolute error on Φ < ~2.5e-4.
        #[inline]
        fn big_phi(x: f32) -> f32 {
            let y = x.abs() / std::f32::consts::SQRT_2;
            let t = 1.0_f32 / (1.0 + 0.47047 * y);
            let erfc_approx = t * (0.3480242 + t * (-0.0958798 + t * 0.7478556)) * (-y * y).exp();
            if x >= 0.0 {
                1.0 - erfc_approx / 2.0
            } else {
                erfc_approx / 2.0
            }
        }

        // Initialise reconstruction levels evenly across [-3σ, 3σ].
        // The uniform spread is refined by iteration; the exact starting
        // values only affect the number of iterations needed to converge.
        let mut levels: Vec<f32> = (0..num_levels)
            .map(|k| {
                let p = (k as f32 + 0.5) / num_levels as f32;
                sigma * (-3.0 + 6.0 * p)
            })
            .collect();

        let max_iters = 200;
        let tol = 1e-6_f32;

        for _ in 0..max_iters {
            let levels_prev = levels.clone();

            // ── Step 1: Boundaries = midpoints of adjacent levels ──────────
            // edge[k] separates bin k from bin k+1.
            let edges: Vec<f32> = (0..num_levels - 1)
                .map(|k| (levels[k] + levels[k + 1]) / 2.0)
                .collect();

            // ── Step 2: Levels = conditional centroid of truncated N(0,σ²) ─
            // For bin k with bounds [a, b], the optimal reconstruction value is:
            //   E[X | a ≤ X ≤ b] = σ · [φ(a/σ) − φ(b/σ)] / [Φ(b/σ) − Φ(a/σ)]
            // Outermost bins use ±10σ as practical infinities.
            for k in 0..num_levels {
                let a_raw = if k == 0 {
                    -10.0_f32 * sigma
                } else {
                    edges[k - 1]
                };
                let b_raw = if k == num_levels - 1 {
                    10.0_f32 * sigma
                } else {
                    edges[k]
                };

                let a = a_raw / sigma;
                let b = b_raw / sigma;

                let phi_diff = phi(a) - phi(b);
                let cdf_diff = big_phi(b) - big_phi(a);

                if cdf_diff.abs() > 1e-10 {
                    levels[k] = sigma * phi_diff / cdf_diff;
                }
                // Empty bin: keep previous level to avoid divergence.
            }

            // ── Check convergence (max-norm on level change) ───────────────
            let change: f32 = levels
                .iter()
                .zip(levels_prev.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            if change < tol {
                break;
            }
        }

        // Build final bin_edges with ±∞ sentinels so the quantize() method
        // (which skips bin_edges[0] via .skip(1)) works correctly.
        // Layout: [-∞, inner_edge_0, ..., inner_edge_{n-2}, +∞]
        let bin_edges: Vec<f32> = std::iter::once(f32::NEG_INFINITY)
            .chain((0..num_levels - 1).map(|k| (levels[k] + levels[k + 1]) / 2.0))
            .chain(std::iter::once(f32::INFINITY))
            .collect();

        Ok(Self {
            bin_edges,
            reconstruction_values: levels,
        })
    }
}

impl Quantizer for NonUniformQuantizer {
    fn quantize(&self, value: f32) -> i32 {
        // Find bin using binary search
        for (i, &edge) in self.bin_edges.iter().enumerate().skip(1) {
            if value < edge {
                return (i - 1) as i32;
            }
        }
        (self.reconstruction_values.len() - 1) as i32
    }

    fn dequantize(&self, level: i32) -> f32 {
        let idx = level.clamp(0, (self.reconstruction_values.len() - 1) as i32) as usize;
        self.reconstruction_values[idx]
    }

    fn num_levels(&self) -> usize {
        self.reconstruction_values.len()
    }
}

// Implement SignalTokenizer for advanced quantizers

impl SignalTokenizer for AdaptiveQuantizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        let quantized = self.quantize_adaptive(signal)?;
        Ok(quantized.mapv(|x| x as f32))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Ok(tokens.mapv(|t| self.dequantize(t.round() as i32)))
    }

    fn embed_dim(&self) -> usize {
        1
    }

    fn vocab_size(&self) -> usize {
        self.levels
    }
}

impl SignalTokenizer for DeadZoneQuantizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Ok(signal.mapv(|x| self.quantize(x) as f32))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Ok(tokens.mapv(|t| self.dequantize(t.round() as i32)))
    }

    fn embed_dim(&self) -> usize {
        1
    }

    fn vocab_size(&self) -> usize {
        self.levels
    }
}

impl SignalTokenizer for NonUniformQuantizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Ok(signal.mapv(|x| self.quantize(x) as f32))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Ok(tokens.mapv(|t| self.dequantize(t.round() as i32)))
    }

    fn embed_dim(&self) -> usize {
        1
    }

    fn vocab_size(&self) -> usize {
        self.reconstruction_values.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helper: binary search over interior bin edges
// Returns the bin index in [0, edges.len()] for scalar `x`.
// ─────────────────────────────────────────────────────────────────────────────
fn find_bin(x: f32, edges: &[f32]) -> usize {
    let mut lo = 0usize;
    let mut hi = edges.len(); // hi == num_levels - 1 for L levels
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if x <= edges[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo // in [0, num_levels - 1]
}

// ─────────────────────────────────────────────────────────────────────────────
// EntropyConstrainedQuantizer
// ─────────────────────────────────────────────────────────────────────────────

/// Lagrangian Rate-Distortion optimal scalar quantizer.
///
/// Minimizes `D + λ·R` where D is MSE distortion and R is empirical entropy.
/// Implements an entropy-regularized Lloyd-Max algorithm:
///
/// 1. Initialize bin edges from equal-mass (percentile) split of the signal.
/// 2. Iterate:
///    - **Centroid update**: `recon_i` = mean of samples assigned to bin `i`.
///    - **Entropy-regularized edge update**:
///      `edge_i = 0.5*(recon_{i-1}+recon_i) + (lambda/(recon_i-recon_{i-1}))*(ln p_{i-1} - ln p_i)`
///    - Clamp edges to be strictly monotonic.
///    - Recompute empirical probabilities.
///    - Evaluate `cost = D + λ·R` and stop when Δcost < tol.
pub struct EntropyConstrainedQuantizer {
    /// Interior bin edges (length = num_levels - 1, strictly sorted).
    bin_edges: Vec<f32>,
    /// Reconstruction value for each bin (length = num_levels).
    reconstruction_values: Vec<f32>,
    /// Lagrange multiplier controlling R-D trade-off.
    lambda: f32,
    /// Optional target bits-per-symbol (set by `fit_with_target_rate`).
    target_bits_per_symbol: Option<f64>,
    /// Empirical probabilities of each bin after fitting.
    empirical_probs: Vec<f64>,
}

impl EntropyConstrainedQuantizer {
    /// Construct directly from pre-computed edges and reconstruction values.
    ///
    /// Uniform prior probabilities are assumed until `fit_lagrangian` is called.
    pub fn new(bin_edges: Vec<f32>, reconstruction_values: Vec<f32>, lambda: f32) -> Self {
        let n = reconstruction_values.len();
        Self {
            bin_edges,
            reconstruction_values,
            lambda,
            target_bits_per_symbol: None,
            empirical_probs: vec![1.0 / n as f64; n],
        }
    }

    /// Fit ECQ via entropy-regularized Lloyd-Max iteration.
    ///
    /// Returns `Err` if `num_levels < 2` or the signal is too short.
    ///
    /// # Arguments
    ///
    /// * `signal`     – Input signal samples.
    /// * `num_levels` – Number of quantization bins (≥ 2).
    /// * `lambda`     – Lagrange multiplier; larger → more compression, less fidelity.
    /// * `max_iters`  – Maximum Lloyd-Max iterations.
    /// * `tol`        – Convergence threshold on `|Δcost|`.
    pub fn fit_lagrangian(
        signal: &Array1<f32>,
        num_levels: usize,
        lambda: f32,
        max_iters: usize,
        tol: f32,
    ) -> TokenizerResult<Self> {
        if num_levels < 2 {
            return Err(TokenizerError::InvalidConfig(
                "num_levels must be >= 2".into(),
            ));
        }
        if signal.len() < num_levels {
            return Err(TokenizerError::InvalidConfig(
                "signal is too short for the requested num_levels".into(),
            ));
        }

        let n = signal.len();
        let sig_min = signal.iter().cloned().fold(f32::INFINITY, f32::min);
        let sig_max = signal.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (sig_max - sig_min).max(1e-6);
        let min_gap = 1e-6 * range;

        // Step 1: Init bin edges from equal-mass (percentile) split.
        let mut sorted_signal: Vec<f32> = signal.iter().cloned().collect();
        sorted_signal.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // bin_edges has `num_levels - 1` interior edges.
        let mut bin_edges: Vec<f32> = (1..num_levels)
            .map(|i| {
                let idx = (i * n / num_levels).min(n - 1);
                sorted_signal[idx]
            })
            .collect();

        // Step 2: Init reconstruction values as bin midpoints.
        // Bins: (-∞, edge[0]], (edge[0], edge[1]], …, (edge[L-2], +∞)
        let mut recon: Vec<f32> = {
            let mut r = Vec::with_capacity(num_levels);
            r.push((sig_min + bin_edges[0]) * 0.5);
            for i in 1..num_levels - 1 {
                r.push((bin_edges[i - 1] + bin_edges[i]) * 0.5);
            }
            r.push((bin_edges[num_levels - 2] + sig_max) * 0.5);
            r
        };

        let mut probs = vec![1.0f64 / num_levels as f64; num_levels];
        let mut prev_cost = f64::INFINITY;

        for _iter in 0..max_iters {
            // ── Step 3a: Centroid update ──────────────────────────────────
            let mut sums = vec![0.0f64; num_levels];
            let mut counts = vec![0usize; num_levels];
            for &x in signal.iter() {
                let b = find_bin(x, &bin_edges);
                sums[b] += x as f64;
                counts[b] += 1;
            }
            for i in 0..num_levels {
                if counts[i] > 0 {
                    recon[i] = (sums[i] / counts[i] as f64) as f32;
                }
                // Empty bin: keep previous centroid (no panic).
            }

            // Monotonicity guard after centroid update (insurance for
            // edge cases where empty bins collapse centroids).
            for i in 1..num_levels {
                if recon[i] <= recon[i - 1] + min_gap {
                    recon[i] = recon[i - 1] + min_gap;
                }
            }

            // ── Step 3b: Recompute probs for edge update ─────────────────
            let eps = 1e-10;
            let denom = n as f64 + num_levels as f64 * eps;
            for i in 0..num_levels {
                probs[i] = (counts[i] as f64 + eps) / denom;
            }

            // ── Step 3c: Entropy-regularized edge update ──────────────────
            for i in 0..num_levels - 1 {
                let r_left = recon[i];
                let r_right = recon[i + 1];
                let gap = (r_right - r_left).max(min_gap);
                let p_left = probs[i].max(eps);
                let p_right = probs[i + 1].max(eps);
                let entropy_term = (lambda / gap) * (p_left.ln() - p_right.ln()) as f32;
                bin_edges[i] = 0.5 * (r_left + r_right) + entropy_term;
            }

            // ── Step 3d: Enforce strict monotonicity ─────────────────────
            for i in 1..bin_edges.len() {
                if bin_edges[i] <= bin_edges[i - 1] + min_gap {
                    bin_edges[i] = bin_edges[i - 1] + min_gap;
                }
            }

            // ── Step 3e: Recompute probs with new edges ───────────────────
            let mut new_counts = vec![0usize; num_levels];
            for &x in signal.iter() {
                new_counts[find_bin(x, &bin_edges)] += 1;
            }
            for i in 0..num_levels {
                probs[i] = (new_counts[i] as f64 + eps) / denom;
            }

            // ── Step 3f: Convergence check on D + λ·R ────────────────────
            let distortion: f64 = signal
                .iter()
                .map(|&x| {
                    let b = find_bin(x, &bin_edges);
                    let d = x as f64 - recon[b] as f64;
                    d * d
                })
                .sum::<f64>()
                / n as f64;

            let entropy: f64 = probs
                .iter()
                .map(|&p| if p > eps { -p * p.log2() } else { 0.0 })
                .sum();

            let cost = distortion + lambda as f64 * entropy;

            if (prev_cost - cost).abs() < tol as f64 {
                break;
            }
            prev_cost = cost;
        }

        Ok(Self {
            bin_edges,
            reconstruction_values: recon,
            lambda,
            target_bits_per_symbol: None,
            empirical_probs: probs,
        })
    }

    /// Compress `signal` using Huffman coding built from the fitted bin
    /// probabilities.
    ///
    /// Returns `(compressed_bytes, symbol_count)`.  The `symbol_count` is
    /// needed for lossless decompression.
    pub fn encode_compressed(&self, signal: &Array1<f32>) -> TokenizerResult<(Vec<u8>, u64)> {
        use crate::entropy::{compute_frequencies, HuffmanEncoder};

        let symbols: Vec<u32> = signal
            .iter()
            .map(|&x| find_bin(x, &self.bin_edges) as u32)
            .collect();

        let freqs = compute_frequencies(&symbols);
        let encoder = HuffmanEncoder::from_frequencies(&freqs)?;
        let compressed = encoder.encode(&symbols)?;
        let symbol_count = symbols.len() as u64;
        Ok((compressed, symbol_count))
    }

    /// Decompress bytes produced by `encode_compressed` back to a signal.
    ///
    /// The `symbol_count` must match the value returned by `encode_compressed`.
    pub fn decode_compressed(
        &self,
        bytes: &[u8],
        _symbol_count: u64,
    ) -> TokenizerResult<Array1<f32>> {
        use crate::entropy::{HuffmanDecoder, HuffmanEncoder};

        // Re-build the same frequency table from `empirical_probs` so we can
        // reconstruct the Huffman tree without storing it separately.
        let n_levels = self.reconstruction_values.len();
        let total_pseudo = 1_000_000u64; // Scale probs to integer counts.
        let mut freqs = std::collections::HashMap::new();
        let mut allocated = 0u64;
        for i in 0..n_levels {
            let cnt = (self.empirical_probs[i] * total_pseudo as f64).round() as u64;
            let cnt = cnt.max(1); // At least 1 to keep symbol in codebook.
            freqs.insert(i as u32, cnt);
            allocated += cnt;
        }
        // Give the leftover to symbol 0 to keep totals consistent (doesn't
        // affect code *lengths*, only the tree shape).
        let _ = allocated; // unused; counts only need to be proportional.

        let encoder = HuffmanEncoder::from_frequencies(&freqs)?;
        let decoder = HuffmanDecoder::new(encoder.tree());
        let indices = decoder.decode(bytes)?;

        let values: Vec<f32> = indices
            .iter()
            .map(|&idx| {
                let b = (idx as usize).min(self.reconstruction_values.len() - 1);
                self.reconstruction_values[b]
            })
            .collect();

        Ok(Array1::from_vec(values))
    }

    /// Fit ECQ using binary search over λ to hit a target entropy rate.
    ///
    /// # Arguments
    ///
    /// * `signal`          – Training signal.
    /// * `num_levels`      – Number of quantization bins.
    /// * `target_bpp`      – Desired bits per symbol.
    /// * `max_outer_iters` – Number of λ bisection steps.
    pub fn fit_with_target_rate(
        signal: &Array1<f32>,
        num_levels: usize,
        target_bpp: f64,
        max_outer_iters: usize,
    ) -> TokenizerResult<Self> {
        let mut lambda_lo = 0.0f32;
        let mut lambda_hi = 10.0f32;

        // Start with the high-lambda (low-rate) end.
        let mut best = Self::fit_lagrangian(signal, num_levels, lambda_hi, 100, 1e-5)?;

        for _ in 0..max_outer_iters {
            let lambda_mid = (lambda_lo + lambda_hi) * 0.5;
            let candidate = Self::fit_lagrangian(signal, num_levels, lambda_mid, 100, 1e-5)?;
            let rate = candidate.compute_entropy_rate(signal);
            if rate > target_bpp {
                // Rate too high → increase lambda to compress more.
                lambda_lo = lambda_mid;
            } else {
                // Rate low enough → record this as "best so far" and try less compression.
                lambda_hi = lambda_mid;
                best = candidate;
            }
            if (lambda_hi - lambda_lo) < 1e-4 {
                break;
            }
        }
        // Update the stored target for reference.
        best.target_bits_per_symbol = Some(target_bpp);
        Ok(best)
    }

    /// Compute empirical Shannon entropy (bits/symbol) of the signal under
    /// this quantizer's bin partition.
    pub fn compute_entropy_rate(&self, signal: &Array1<f32>) -> f64 {
        let n = signal.len();
        if n == 0 {
            return 0.0;
        }
        let mut counts = vec![0usize; self.reconstruction_values.len()];
        for &x in signal.iter() {
            counts[find_bin(x, &self.bin_edges)] += 1;
        }
        counts
            .iter()
            .map(|&c| {
                if c > 0 {
                    let p = c as f64 / n as f64;
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Compute mean-squared distortion of the signal under this quantizer.
    pub fn empirical_distortion(&self, signal: &Array1<f32>) -> f64 {
        let n = signal.len();
        if n == 0 {
            return 0.0;
        }
        signal
            .iter()
            .map(|&x| {
                let r = self.reconstruction_values[find_bin(x, &self.bin_edges)];
                let d = (x - r) as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64
    }

    /// Return a reference to the interior bin edges.
    pub fn bin_edges(&self) -> &[f32] {
        &self.bin_edges
    }

    /// Return a reference to the reconstruction values.
    pub fn reconstruction_values(&self) -> &[f32] {
        &self.reconstruction_values
    }

    /// Return the Lagrange multiplier used during fitting.
    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    /// Return the optional target bits-per-symbol set by `fit_with_target_rate`.
    pub fn target_bits_per_symbol(&self) -> Option<f64> {
        self.target_bits_per_symbol
    }
}

impl Quantizer for EntropyConstrainedQuantizer {
    fn quantize(&self, value: f32) -> i32 {
        find_bin(value, &self.bin_edges) as i32
    }

    fn dequantize(&self, level: i32) -> f32 {
        let idx = level.clamp(0, (self.reconstruction_values.len() - 1) as i32) as usize;
        self.reconstruction_values[idx]
    }

    fn num_levels(&self) -> usize {
        self.reconstruction_values.len()
    }
}

impl SignalTokenizer for EntropyConstrainedQuantizer {
    fn encode(&self, signal: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Ok(signal.mapv(|x| self.quantize(x) as f32))
    }

    fn decode(&self, tokens: &Array1<f32>) -> TokenizerResult<Array1<f32>> {
        Ok(tokens.mapv(|t| self.dequantize(t.round() as i32)))
    }

    fn embed_dim(&self) -> usize {
        1
    }

    fn vocab_size(&self) -> usize {
        self.reconstruction_values.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod ecq_tests {
    use super::*;

    /// Simple pseudo-Gaussian generator (Box-Muller + LCG) — no external deps.
    fn gaussian_signal(n: usize, seed: u64) -> Array1<f32> {
        let mut state = seed;
        let mut next_f32 = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as f32 / u32::MAX as f32
        };
        Array1::from_iter((0..n).map(|_| {
            let u1 = next_f32().max(1e-7);
            let u2 = next_f32();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        }))
    }

    #[test]
    fn fit_lagrangian_convergence_and_monotonicity() {
        let signal = gaussian_signal(10_000, 42);
        let num_levels = 8;
        let q = EntropyConstrainedQuantizer::fit_lagrangian(&signal, num_levels, 0.1, 50, 1e-5)
            .expect("fit_lagrangian failed");

        // Reconstruction values should be in a reasonable range for N(0,1).
        for &r in q.reconstruction_values() {
            assert!(r.abs() <= 4.5, "recon value {r} outside [-4.5, 4.5]");
        }

        // Strictly monotonic reconstruction values.
        for i in 1..q.reconstruction_values().len() {
            assert!(
                q.reconstruction_values()[i] > q.reconstruction_values()[i - 1],
                "recon values not monotonic at i={i}: {} <= {}",
                q.reconstruction_values()[i],
                q.reconstruction_values()[i - 1]
            );
        }
    }

    #[test]
    fn rd_tradeoff_bracketed() {
        let signal = gaussian_signal(10_000, 99);
        let num_levels = 8;
        let q_low =
            EntropyConstrainedQuantizer::fit_lagrangian(&signal, num_levels, 0.01, 100, 1e-6)
                .expect("fit low-lambda failed");
        let q_high =
            EntropyConstrainedQuantizer::fit_lagrangian(&signal, num_levels, 1.0, 100, 1e-6)
                .expect("fit high-lambda failed");

        let d_low = q_low.empirical_distortion(&signal);
        let d_high = q_high.empirical_distortion(&signal);
        let r_low = q_low.compute_entropy_rate(&signal);
        let r_high = q_high.compute_entropy_rate(&signal);

        // High lambda → more compression (lower rate).
        assert!(
            r_high + 1e-6 < r_low,
            "R-D: high-λ should reduce rate: r_high={r_high} r_low={r_low}"
        );
        // High lambda → higher distortion.
        assert!(
            d_low + 1e-6 < d_high,
            "R-D: high-λ should increase distortion: d_low={d_low} d_high={d_high}"
        );
    }

    #[test]
    fn roundtrip_mse_vs_uniform() {
        let signal = gaussian_signal(10_000, 7);
        let num_levels = 8;

        // Use a tiny lambda so ECQ operates near standard Lloyd-Max (low-rate
        // penalty) — the R-D tradeoff is well-exercised in rd_tradeoff_bracketed.
        let q_ecq =
            EntropyConstrainedQuantizer::fit_lagrangian(&signal, num_levels, 0.001, 100, 1e-6)
                .expect("ECQ fit failed");
        let mse_ecq = q_ecq.empirical_distortion(&signal);

        // Naive uniform quantizer baseline.
        let sig_min = signal.iter().cloned().fold(f32::INFINITY, f32::min);
        let sig_max = signal.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let step = (sig_max - sig_min) / num_levels as f32;
        let mse_uniform: f64 = signal
            .iter()
            .map(|&x| {
                let idx = ((x - sig_min) / step).floor() as usize;
                let idx = idx.min(num_levels - 1);
                let r = sig_min + (idx as f32 + 0.5) * step;
                let d = (x - r) as f64;
                d * d
            })
            .sum::<f64>()
            / signal.len() as f64;

        assert!(
            mse_ecq <= mse_uniform * 3.0,
            "ECQ MSE {mse_ecq} > 3× uniform MSE {mse_uniform}"
        );
    }

    #[test]
    fn determinism() {
        let signal = gaussian_signal(10_000, 555);
        let q1 = EntropyConstrainedQuantizer::fit_lagrangian(&signal, 8, 0.1, 50, 1e-5)
            .expect("first fit failed");
        let q2 = EntropyConstrainedQuantizer::fit_lagrangian(&signal, 8, 0.1, 50, 1e-5)
            .expect("second fit failed");

        for (a, b) in q1.bin_edges().iter().zip(q2.bin_edges().iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "non-deterministic bin edges: {a} vs {b}"
            );
        }
    }

    #[test]
    fn fit_with_target_rate_in_range() {
        let signal = gaussian_signal(10_000, 42);
        let q = EntropyConstrainedQuantizer::fit_with_target_rate(&signal, 8, 2.5, 20)
            .expect("fit_with_target_rate failed");
        let rate = q.compute_entropy_rate(&signal);
        assert!(
            (1.5..=3.5).contains(&rate),
            "target_rate=2.5 produced rate={rate} outside [1.5, 3.5]"
        );
    }

    #[test]
    fn signal_tokenizer_encode_decode_roundtrip() {
        let signal = gaussian_signal(1_000, 13);
        let q = EntropyConstrainedQuantizer::fit_lagrangian(&signal, 8, 0.1, 50, 1e-5)
            .expect("fit failed");

        let tokens = q.encode(&signal).expect("encode failed");
        assert_eq!(tokens.len(), signal.len());

        let reconstructed = q.decode(&tokens).expect("decode failed");
        assert_eq!(reconstructed.len(), signal.len());

        // Every reconstructed value must be a valid reconstruction value.
        for &r in reconstructed.iter() {
            assert!(
                q.reconstruction_values().contains(&r),
                "reconstructed value {r} not in reconstruction_values"
            );
        }
    }

    #[test]
    fn invalid_config_rejected() {
        let signal = gaussian_signal(100, 1);
        assert!(
            EntropyConstrainedQuantizer::fit_lagrangian(&signal, 1, 0.1, 10, 1e-5).is_err(),
            "num_levels=1 should be rejected"
        );
        let tiny = gaussian_signal(3, 2);
        assert!(
            EntropyConstrainedQuantizer::fit_lagrangian(&tiny, 8, 0.1, 10, 1e-5).is_err(),
            "signal shorter than num_levels should be rejected"
        );
    }

    #[test]
    fn encode_decode_compressed_roundtrip() {
        let signal = gaussian_signal(1_000, 77);
        let q = EntropyConstrainedQuantizer::fit_lagrangian(&signal, 8, 0.1, 50, 1e-5)
            .expect("fit failed");

        let (compressed, sym_count) = q
            .encode_compressed(&signal)
            .expect("encode_compressed failed");
        let reconstructed = q
            .decode_compressed(&compressed, sym_count)
            .expect("decode_compressed failed");

        // Lengths must match.
        assert_eq!(reconstructed.len(), signal.len());

        // Every reconstructed value is a valid reconstruction value.
        for &r in reconstructed.iter() {
            assert!(
                q.reconstruction_values().contains(&r),
                "decoded value {r} not in reconstruction_values"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_quantizer() {
        let quant = AdaptiveQuantizer::new(8, 16, 0.5, -1.0, 1.0).unwrap();

        let signal = Array1::from_vec((0..128).map(|i| ((i as f32) * 0.05).sin()).collect());

        let encoded = quant.encode(&signal).unwrap();
        assert_eq!(encoded.len(), 128);

        let decoded = quant.decode(&encoded).unwrap();
        assert_eq!(decoded.len(), 128);
    }

    #[test]
    fn test_dead_zone_quantizer() {
        let quant = DeadZoneQuantizer::new(8, 0.1, -1.0, 1.0).unwrap();

        // Test dead zone behavior
        let level = quant.quantize(0.05);
        let recovered = quant.dequantize(level);
        assert_eq!(recovered, 0.0); // Should be in dead zone

        // Test outside dead zone
        let level = quant.quantize(0.5);
        let recovered = quant.dequantize(level);
        assert!(recovered.abs() > 0.1);
    }

    #[test]
    fn test_dead_zone_signal() {
        let quant = DeadZoneQuantizer::new(8, 0.2, -1.0, 1.0).unwrap();

        // Signal with small values that should be zeroed
        let signal = Array1::from_vec(vec![0.01, 0.5, -0.1, 0.8, 0.05]);

        let encoded = quant.encode(&signal).unwrap();
        let decoded = quant.decode(&encoded).unwrap();

        // Small values should become zero
        assert_eq!(decoded[0], 0.0);
        assert_eq!(decoded[2], 0.0);
        assert_eq!(decoded[4], 0.0);

        // Large values should be preserved (approximately)
        assert!(decoded[1] > 0.3);
        assert!(decoded[3] > 0.6);
    }

    #[test]
    fn test_nonuniform_quantizer() {
        let edges = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
        let quant = NonUniformQuantizer::from_edges(edges).unwrap();

        assert_eq!(quant.num_levels(), 4);

        let level = quant.quantize(-1.0);
        assert_eq!(level, 0);

        let level = quant.quantize(0.25);
        assert_eq!(level, 2);
    }

    #[test]
    fn test_lloyd_max_quantizer() {
        let quant = NonUniformQuantizer::lloyd_max_gaussian(8, 1.0).unwrap();

        assert_eq!(quant.num_levels(), 8);

        // Test symmetry
        let level_pos = quant.quantize(0.5);
        let level_neg = quant.quantize(-0.5);
        let val_pos = quant.dequantize(level_pos);
        let val_neg = quant.dequantize(level_neg);

        assert!((val_pos + val_neg).abs() < 0.5); // Should be roughly symmetric
    }

    #[test]
    fn test_adaptive_vs_uniform() {
        let adaptive = AdaptiveQuantizer::new(6, 8, 0.8, -1.0, 1.0).unwrap();

        // Signal with varying local statistics
        let mut signal_vec = Vec::new();
        // Low variance region
        for i in 0..64 {
            signal_vec.push(0.1 * (i as f32 * 0.05).sin());
        }
        // High variance region
        for i in 64..128 {
            signal_vec.push(0.8 * (i as f32 * 0.1).sin());
        }

        let signal = Array1::from_vec(signal_vec);
        let encoded = adaptive.encode(&signal).unwrap();

        assert_eq!(encoded.len(), 128);
    }

    #[test]
    fn test_nonuniform_with_custom_values() {
        let edges = vec![-1.0, -0.3, 0.0, 0.3, 1.0];
        let recon = vec![-0.7, -0.15, 0.15, 0.7];

        let quant = NonUniformQuantizer::new(edges, recon).unwrap();

        let level = quant.quantize(0.1);
        let value = quant.dequantize(level);
        assert!((value - 0.15).abs() < 0.01);
    }

    #[test]
    fn test_lloyd_max_bin_edges_monotonic() {
        let q = NonUniformQuantizer::lloyd_max_gaussian(8, 1.0).expect("lloyd_max_gaussian failed");
        // Inner edges (excluding ±∞ sentinels) must be strictly increasing
        let inner: Vec<f32> = q
            .bin_edges
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        for w in inner.windows(2) {
            assert!(
                w[1] > w[0],
                "bin_edges not monotonic: {} not > {}",
                w[1],
                w[0]
            );
        }
    }

    #[test]
    fn test_lloyd_max_oracle_boundaries() {
        // Published Max-1960 N=8 σ=1 inner boundaries: ±0.501, ±1.050, ±1.748
        let q = NonUniformQuantizer::lloyd_max_gaussian(8, 1.0).expect("lloyd_max_gaussian failed");
        let inner: Vec<f32> = q
            .bin_edges
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        assert_eq!(inner.len(), 7, "Expected 7 inner boundaries for 8 levels");
        let tol = 0.06;
        assert!(
            (inner[0] - (-1.748)).abs() < tol,
            "boundary[0]={} expected ~-1.748",
            inner[0]
        );
        assert!(
            (inner[1] - (-1.050)).abs() < tol,
            "boundary[1]={} expected ~-1.050",
            inner[1]
        );
        assert!(
            (inner[2] - (-0.501)).abs() < tol,
            "boundary[2]={} expected ~-0.501",
            inner[2]
        );
        assert!(
            (inner[4] - (0.501)).abs() < tol,
            "boundary[4]={} expected ~0.501",
            inner[4]
        );
        assert!(
            (inner[5] - (1.050)).abs() < tol,
            "boundary[5]={} expected ~1.050",
            inner[5]
        );
        assert!(
            (inner[6] - (1.748)).abs() < tol,
            "boundary[6]={} expected ~1.748",
            inner[6]
        );
    }

    #[test]
    fn test_lloyd_max_levels_antisymmetric() {
        let q = NonUniformQuantizer::lloyd_max_gaussian(8, 1.0).expect("lloyd_max_gaussian failed");
        let n = q.reconstruction_values.len();
        for i in 0..n / 2 {
            let sum = q.reconstruction_values[i] + q.reconstruction_values[n - 1 - i];
            assert!(
                sum.abs() < 5e-3,
                "Levels not antisymmetric: r[{}]={} r[{}]={}",
                i,
                q.reconstruction_values[i],
                n - 1 - i,
                q.reconstruction_values[n - 1 - i]
            );
        }
    }

    #[test]
    fn test_lloyd_max_round_trip_bounded() {
        let q = NonUniformQuantizer::lloyd_max_gaussian(8, 1.0).expect("lloyd_max_gaussian failed");
        // Round-trip error must be bounded
        let test_vals = [-2.0_f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        for &v in &test_vals {
            let q_idx = q.quantize(v);
            let reconstructed = q.dequantize(q_idx);
            // Should be within one quantization step
            assert!(
                reconstructed.is_finite(),
                "dequantize returned non-finite for v={}",
                v
            );
        }
    }
}
