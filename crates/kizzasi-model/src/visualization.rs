//! State Visualization and Attention Pattern Analysis
//!
//! Provides tools for visualizing internal SSM states, gating patterns,
//! and phase portraits — all in pure Rust with no external plotting dependencies.
//!
//! # Overview
//!
//! - [`ActivationHistogram`]: 2D histogram of activation distributions across dims
//! - [`GatingPatternRecorder`]: Records and analyzes SSM gating weights over time
//! - [`PhasePortrait`]: PCA projection, divergence, fixed-point and periodicity analysis
//! - [`matrix_to_csv`]: Export Array2 to CSV-formatted String
//! - [`signal_to_svg_sparkline`]: Export 1D signal as inline SVG sparkline

use crate::error::{ModelError, ModelResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// ActivationHistogram
// ---------------------------------------------------------------------------

/// 2D histogram of activation distributions across sequence steps.
///
/// Bins are computed along the value axis; each column represents one
/// dimension of the activation vector.
#[derive(Debug, Clone)]
pub struct ActivationHistogram {
    /// Bin counts — shape `(num_bins, num_dims)`
    pub bins: Array2<f32>,
    /// Bin edges — shape `(num_bins + 1,)`
    pub edges: Array1<f32>,
    /// Number of activation dimensions
    pub num_dims: usize,
    num_bins: usize,
    total_counts: Array1<f32>,
}

impl ActivationHistogram {
    /// Create a new histogram with `num_bins` evenly spaced between `min_val`
    /// and `max_val` for `num_dims` dimensions.
    pub fn new(num_bins: usize, min_val: f32, max_val: f32, num_dims: usize) -> Self {
        let bins = Array2::zeros((num_bins, num_dims));
        let step = (max_val - min_val) / num_bins as f32;
        let edges = Array1::from_vec((0..=num_bins).map(|i| min_val + i as f32 * step).collect());
        let total_counts = Array1::zeros(num_dims);
        Self {
            bins,
            edges,
            num_dims,
            num_bins,
            total_counts,
        }
    }

    /// Accumulate one activation vector into the histogram.
    pub fn update(&mut self, x: &Array1<f32>) -> ModelResult<()> {
        if x.len() != self.num_dims {
            return Err(ModelError::dimension_mismatch(
                "ActivationHistogram::update",
                self.num_dims,
                x.len(),
            ));
        }
        let min_val = self.edges[0];
        let max_val = self.edges[self.num_bins];
        let range = max_val - min_val;
        if range <= 0.0 {
            return Err(ModelError::invalid_config(
                "ActivationHistogram: zero-range edges",
            ));
        }
        for (d, &v) in x.iter().enumerate() {
            // Clamp to [min, max) then find bin
            let clamped = v.clamp(min_val, max_val - f32::EPSILON * range);
            let frac = (clamped - min_val) / range;
            let bin = (frac * self.num_bins as f32) as usize;
            let bin = bin.min(self.num_bins - 1);
            self.bins[(bin, d)] += 1.0;
            self.total_counts[d] += 1.0;
        }
        Ok(())
    }

    /// Return normalized histogram (density) per dimension.
    ///
    /// Each column sums to ≈ 1.0 (or 0.0 if no data was accumulated).
    pub fn density(&self) -> Array2<f32> {
        let mut out = Array2::zeros((self.num_bins, self.num_dims));
        for d in 0..self.num_dims {
            let total = self.total_counts[d];
            if total > 0.0 {
                for b in 0..self.num_bins {
                    out[(b, d)] = self.bins[(b, d)] / total;
                }
            }
        }
        out
    }

    /// Per-dimension Shannon entropy of the distribution (in nats).
    pub fn per_dim_entropy(&self) -> Array1<f32> {
        let density = self.density();
        let mut entropy = Array1::zeros(self.num_dims);
        for d in 0..self.num_dims {
            let mut h = 0.0_f32;
            for b in 0..self.num_bins {
                let p = density[(b, d)];
                if p > 0.0 {
                    h -= p * p.ln();
                }
            }
            entropy[d] = h;
        }
        entropy
    }

    /// Return indices of the `top_k` dimensions with highest entropy.
    pub fn most_active_dims(&self, top_k: usize) -> Vec<usize> {
        let entropy = self.per_dim_entropy();
        let mut indexed: Vec<(usize, f32)> = entropy.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = top_k.min(self.num_dims);
        indexed.into_iter().take(k).map(|(i, _)| i).collect()
    }
}

// ---------------------------------------------------------------------------
// GatingPatternRecorder
// ---------------------------------------------------------------------------

/// Records SSM gating weights over time and provides cross-correlation analysis.
#[derive(Debug, Clone)]
pub struct GatingPatternRecorder {
    patterns: Vec<Array1<f32>>,
    max_steps: usize,
}

impl GatingPatternRecorder {
    /// Create a recorder that stores up to `max_steps` gating vectors.
    pub fn new(max_steps: usize) -> Self {
        Self {
            patterns: Vec::with_capacity(max_steps),
            max_steps,
        }
    }

    /// Record one gating vector. Returns an error if `max_steps` is already reached.
    pub fn record(&mut self, gate: &Array1<f32>) -> ModelResult<()> {
        if self.patterns.len() >= self.max_steps {
            return Err(ModelError::invalid_config(
                "GatingPatternRecorder: max_steps exceeded",
            ));
        }
        if !self.patterns.is_empty() && gate.len() != self.patterns[0].len() {
            return Err(ModelError::dimension_mismatch(
                "GatingPatternRecorder::record",
                self.patterns[0].len(),
                gate.len(),
            ));
        }
        self.patterns.push(gate.clone());
        Ok(())
    }

    /// Return the recorded patterns as a `(T, D)` matrix.
    pub fn as_matrix(&self) -> ModelResult<Array2<f32>> {
        if self.patterns.is_empty() {
            return Err(ModelError::invalid_config(
                "GatingPatternRecorder: no patterns recorded",
            ));
        }
        let t = self.patterns.len();
        let d = self.patterns[0].len();
        let mut m = Array2::zeros((t, d));
        for (i, p) in self.patterns.iter().enumerate() {
            for (j, &v) in p.iter().enumerate() {
                m[(i, j)] = v;
            }
        }
        Ok(m)
    }

    /// Compute the `(D, D)` Pearson cross-correlation matrix between dimensions.
    ///
    /// Diagonal entries are 1.0 (self-correlation). For zero-variance dims,
    /// correlation is set to 0.0.
    pub fn cross_correlation(&self) -> ModelResult<Array2<f32>> {
        let m = self.as_matrix()?;
        let t = m.nrows();
        let d = m.ncols();

        if t < 2 {
            return Err(ModelError::invalid_config(
                "GatingPatternRecorder::cross_correlation: need at least 2 time steps",
            ));
        }

        // Per-dimension mean and std
        let mut means = Array1::<f32>::zeros(d);
        let mut stds = Array1::<f32>::zeros(d);
        for j in 0..d {
            let sum: f32 = (0..t).map(|i| m[(i, j)]).sum();
            let mean = sum / t as f32;
            means[j] = mean;
            let var: f32 = (0..t).map(|i| (m[(i, j)] - mean).powi(2)).sum::<f32>() / t as f32;
            stds[j] = var.sqrt();
        }

        let mut corr = Array2::<f32>::zeros((d, d));
        for a in 0..d {
            for b in 0..d {
                if stds[a] < 1e-12 || stds[b] < 1e-12 {
                    // Zero-variance dimension: 1.0 on diagonal, 0.0 elsewhere
                    corr[(a, b)] = if a == b { 1.0 } else { 0.0 };
                } else {
                    let cov: f32 = (0..t)
                        .map(|i| (m[(i, a)] - means[a]) * (m[(i, b)] - means[b]))
                        .sum::<f32>()
                        / t as f32;
                    corr[(a, b)] = (cov / (stds[a] * stds[b])).clamp(-1.0, 1.0);
                }
            }
        }
        Ok(corr)
    }

    /// Return `(dim_a, dim_b, correlation)` pairs where |correlation| >= threshold.
    pub fn correlated_dims(&self, threshold: f32) -> ModelResult<Vec<(usize, usize, f32)>> {
        let corr = self.cross_correlation()?;
        let d = corr.nrows();
        let mut result = Vec::new();
        for a in 0..d {
            for b in (a + 1)..d {
                let c = corr[(a, b)];
                if c.abs() >= threshold {
                    result.push((a, b, c));
                }
            }
        }
        Ok(result)
    }

    /// Return a smoothed version of the pattern matrix (causal moving average over time).
    pub fn smoothed(&self, window: usize) -> ModelResult<Array2<f32>> {
        let m = self.as_matrix()?;
        let t = m.nrows();
        let d = m.ncols();
        let w = window.max(1);
        let mut out = Array2::zeros((t, d));
        for i in 0..t {
            let start = (i + 1).saturating_sub(w);
            let count = (i - start + 1) as f32;
            for j in 0..d {
                let sum: f32 = (start..=i).map(|k| m[(k, j)]).sum();
                out[(i, j)] = sum / count;
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// PhasePortrait
// ---------------------------------------------------------------------------

/// SSM state trajectory visualizer with phase portrait, PCA projection,
/// and attractor analysis.
#[derive(Debug, Clone)]
pub struct PhasePortrait {
    trajectory: Vec<Array1<f32>>,
    dim: usize,
}

impl PhasePortrait {
    /// Create a new phase portrait buffer of the given `dim`, pre-allocated
    /// for `capacity` steps.
    pub fn new(dim: usize, capacity: usize) -> Self {
        Self {
            trajectory: Vec::with_capacity(capacity),
            dim,
        }
    }

    /// Record a hidden state vector.
    pub fn record(&mut self, state: &Array1<f32>) -> ModelResult<()> {
        if state.len() != self.dim {
            return Err(ModelError::dimension_mismatch(
                "PhasePortrait::record",
                self.dim,
                state.len(),
            ));
        }
        self.trajectory.push(state.clone());
        Ok(())
    }

    /// Project trajectory onto its top-2 principal components.
    ///
    /// Uses pure-Rust power iteration with deflation. Returns `(T, 2)` matrix.
    pub fn pca_projection(&self) -> ModelResult<Array2<f32>> {
        let t = self.trajectory.len();
        if t < 2 {
            return Err(ModelError::invalid_config(
                "PhasePortrait::pca_projection: need at least 2 recorded states",
            ));
        }
        let d = self.dim;

        // Build centered data matrix X: (T, D)
        let mut data = Array2::<f32>::zeros((t, d));
        for (i, s) in self.trajectory.iter().enumerate() {
            for (j, &v) in s.iter().enumerate() {
                data[(i, j)] = v;
            }
        }
        // Center
        for j in 0..d {
            let col_mean: f32 = (0..t).map(|i| data[(i, j)]).sum::<f32>() / t as f32;
            for i in 0..t {
                data[(i, j)] -= col_mean;
            }
        }

        let mut out = Array2::<f32>::zeros((t, 2));

        // Compute top-2 principal components via power iteration + deflation
        let mut data_copy = data.clone();
        for pc_idx in 0..2 {
            // Initialise direction vector (deterministic: unit vector along axis 0)
            let mut v = Array1::<f32>::zeros(d);
            v[pc_idx % d] = 1.0;

            for _ in 0..50 {
                // u = X v  (T,)
                let mut u = Array1::<f32>::zeros(t);
                for i in 0..t {
                    u[i] = (0..d).map(|j| data_copy[(i, j)] * v[j]).sum();
                }
                // v_new = X^T u  (D,)
                let mut v_new = Array1::<f32>::zeros(d);
                for j in 0..d {
                    v_new[j] = (0..t).map(|i| data_copy[(i, j)] * u[i]).sum();
                }
                // Normalize
                let norm = v_new.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm < 1e-12 {
                    break;
                }
                v = v_new.mapv(|x| x / norm);
            }

            // Project: scores = X v  (T,)
            for i in 0..t {
                let proj: f32 = (0..d).map(|j| data_copy[(i, j)] * v[j]).sum();
                out[(i, pc_idx)] = proj;
            }

            // Deflate: X = X - scores * v^T
            for i in 0..t {
                let score = out[(i, pc_idx)];
                for j in 0..d {
                    data_copy[(i, j)] -= score * v[j];
                }
            }
        }

        Ok(out)
    }

    /// Lyapunov-like divergence estimate: average log ratio of consecutive distances.
    ///
    /// Returns the mean of log(d(t+1) / d(t)) where d(t) = ||state(t+1) - state(t)||.
    pub fn divergence_estimate(&self) -> ModelResult<f32> {
        let t = self.trajectory.len();
        if t < 3 {
            return Err(ModelError::invalid_config(
                "PhasePortrait::divergence_estimate: need at least 3 states",
            ));
        }
        let mut log_ratios = Vec::new();
        let dist = |a: &Array1<f32>, b: &Array1<f32>| -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        };
        for i in 0..(t - 2) {
            let d0 = dist(&self.trajectory[i], &self.trajectory[i + 1]);
            let d1 = dist(&self.trajectory[i + 1], &self.trajectory[i + 2]);
            if d0 > 1e-12 && d1 > 1e-12 {
                log_ratios.push((d1 / d0).ln());
            }
        }
        if log_ratios.is_empty() {
            return Ok(0.0);
        }
        Ok(log_ratios.iter().sum::<f32>() / log_ratios.len() as f32)
    }

    /// Detect fixed points: states that recur within `tolerance` (L2).
    ///
    /// Returns one representative state per cluster.
    pub fn fixed_points(&self, tolerance: f32) -> Vec<Array1<f32>> {
        let mut representatives: Vec<Array1<f32>> = Vec::new();
        let dist = |a: &Array1<f32>, b: &Array1<f32>| -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        };
        for state in &self.trajectory {
            let already_covered = representatives
                .iter()
                .any(|rep| dist(rep, state) <= tolerance);
            if !already_covered {
                representatives.push(state.clone());
            }
        }
        representatives
    }

    /// Estimate periodicity via the peak autocorrelation of the trajectory's
    /// norm sequence (excluding lag-0).
    ///
    /// Returns a value in `[0, 1]` where higher means more periodic.
    pub fn periodicity_score(&self) -> ModelResult<f32> {
        let t = self.trajectory.len();
        if t < 4 {
            return Err(ModelError::invalid_config(
                "PhasePortrait::periodicity_score: need at least 4 states",
            ));
        }

        // Compute norm sequence
        let norms: Vec<f32> = self
            .trajectory
            .iter()
            .map(|s| s.iter().map(|&x| x * x).sum::<f32>().sqrt())
            .collect();

        let mean = norms.iter().sum::<f32>() / t as f32;
        let centered: Vec<f32> = norms.iter().map(|&x| x - mean).collect();
        let var: f32 = centered.iter().map(|&x| x * x).sum::<f32>() / t as f32;

        if var < 1e-12 {
            // Constant sequence — perfectly periodic (or trivially so)
            return Ok(1.0);
        }

        // Compute autocorrelation for lags 1 .. t/2
        let max_lag = (t / 2).max(1);
        let mut peak = 0.0_f32;
        for lag in 1..=max_lag {
            let cov: f32 = (0..(t - lag))
                .map(|i| centered[i] * centered[i + lag])
                .sum::<f32>()
                / (t - lag) as f32;
            let acf = (cov / var).abs();
            if acf > peak {
                peak = acf;
            }
        }
        Ok(peak.min(1.0))
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Export a `(rows, cols)` matrix to CSV-like text.
///
/// Each row becomes one line; values are separated by commas.
/// No file I/O is performed — the result is returned as a `String`.
pub fn matrix_to_csv(m: &Array2<f32>) -> String {
    let (rows, cols) = m.dim();
    let mut lines = Vec::with_capacity(rows);
    for i in 0..rows {
        let row_str: Vec<String> = (0..cols).map(|j| format!("{}", m[(i, j)])).collect();
        lines.push(row_str.join(","));
    }
    lines.join("\n")
}

/// Render a 1D signal as an inline SVG sparkline.
///
/// Returns a complete `<svg>...</svg>` string with a `<polyline>` tracing the
/// signal. No external crates or file I/O are used.
pub fn signal_to_svg_sparkline(signal: &Array1<f32>, width: usize, height: usize) -> String {
    let n = signal.len();
    if n == 0 || width == 0 || height == 0 {
        return format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}"></svg>"#,
            width, height
        );
    }

    let min_val = signal.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = signal.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_val - min_val).max(f32::EPSILON);

    let pad = 2usize;
    let draw_w = (width.saturating_sub(pad * 2)).max(1) as f32;
    let draw_h = (height.saturating_sub(pad * 2)).max(1) as f32;

    let points: Vec<String> = signal
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = pad as f32 + i as f32 * draw_w / (n - 1).max(1) as f32;
            // SVG y-axis is top-down, so invert
            let y = pad as f32 + (1.0 - (v - min_val) / range) * draw_h;
            format!("{:.2},{:.2}", x, y)
        })
        .collect();

    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><polyline points="{pts}" fill="none" stroke="#4488cc" stroke-width="1.5"/></svg>"##,
        w = width,
        h = height,
        pts = points.join(" ")
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: deterministic "random" f32 in [0, 1) from an index seed.
    fn pseudo_rand(seed: usize) -> f32 {
        // LCG
        let v = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (v & 0xFFFF) as f32 / 65536.0
    }

    #[test]
    fn test_histogram_update_and_density() {
        let num_bins = 10;
        let num_dims = 4;
        let mut hist = ActivationHistogram::new(num_bins, -1.0, 1.0, num_dims);

        for i in 0..100 {
            let vals: Vec<f32> = (0..num_dims)
                .map(|d| pseudo_rand(i * num_dims + d) * 2.0 - 1.0)
                .collect();
            let x = Array1::from_vec(vals);
            hist.update(&x).expect("update failed");
        }

        let density = hist.density();
        assert_eq!(density.nrows(), num_bins);
        assert_eq!(density.ncols(), num_dims);

        for d in 0..num_dims {
            let col_sum: f32 = (0..num_bins).map(|b| density[(b, d)]).sum();
            assert!(
                (col_sum - 1.0).abs() < 1e-3,
                "density sum for dim {d} = {col_sum}"
            );
        }
    }

    #[test]
    fn test_histogram_most_active_dims() {
        let num_bins = 8;
        let num_dims = 6;
        let mut hist = ActivationHistogram::new(num_bins, 0.0, 1.0, num_dims);

        for i in 0..80 {
            let vals: Vec<f32> = (0..num_dims)
                .map(|d| pseudo_rand(i * num_dims + d + 1))
                .collect();
            hist.update(&Array1::from_vec(vals)).expect("update failed");
        }

        let top2 = hist.most_active_dims(2);
        assert_eq!(top2.len(), 2);
        for &idx in &top2 {
            assert!(idx < num_dims);
        }
        // The two returned dims should be distinct
        assert_ne!(top2[0], top2[1]);
    }

    #[test]
    fn test_gating_pattern_as_matrix() {
        let dim = 8;
        let steps = 20;
        let mut recorder = GatingPatternRecorder::new(50);

        for i in 0..steps {
            let gate = Array1::from_vec((0..dim).map(|d| pseudo_rand(i * dim + d)).collect());
            recorder.record(&gate).expect("record failed");
        }

        let m = recorder.as_matrix().expect("as_matrix failed");
        assert_eq!(m.nrows(), steps);
        assert_eq!(m.ncols(), dim);
    }

    #[test]
    fn test_gating_pattern_cross_correlation_diagonal() {
        let dim = 4;
        let steps = 30;
        let mut recorder = GatingPatternRecorder::new(100);

        for i in 0..steps {
            let gate = Array1::from_vec((0..dim).map(|d| pseudo_rand(i * dim + d + 42)).collect());
            recorder.record(&gate).expect("record failed");
        }

        let corr = recorder
            .cross_correlation()
            .expect("cross_correlation failed");
        assert_eq!(corr.nrows(), dim);
        assert_eq!(corr.ncols(), dim);

        for d in 0..dim {
            let diag = corr[(d, d)];
            assert!(
                (diag - 1.0).abs() < 1e-4,
                "diagonal[{d}] = {diag}, expected ≈ 1.0"
            );
        }
    }

    #[test]
    fn test_phase_portrait_pca_projection() {
        let dim = 16;
        let steps = 30;
        let mut pp = PhasePortrait::new(dim, 64);

        for i in 0..steps {
            let state = Array1::from_vec(
                (0..dim)
                    .map(|d| pseudo_rand(i * dim + d + 7) * 2.0 - 1.0)
                    .collect(),
            );
            pp.record(&state).expect("record failed");
        }

        let proj = pp.pca_projection().expect("pca_projection failed");
        assert_eq!(proj.nrows(), steps);
        assert_eq!(proj.ncols(), 2);
    }

    #[test]
    fn test_phase_portrait_fixed_points() {
        let dim = 4;
        let mut pp = PhasePortrait::new(dim, 20);
        let fixed = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        for _ in 0..10 {
            pp.record(&fixed).expect("record failed");
        }

        let fps = pp.fixed_points(1e-3);
        assert_eq!(
            fps.len(),
            1,
            "expected exactly 1 fixed point, got {}",
            fps.len()
        );
        // The representative should match the recorded state
        for (a, b) in fps[0].iter().zip(fixed.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matrix_to_csv_format() {
        let m = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("shape error");

        let csv = matrix_to_csv(&m);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3, "expected 3 lines");

        for line in &lines {
            let comma_count = line.chars().filter(|&c| c == ',').count();
            assert_eq!(
                comma_count, 3,
                "expected 3 commas per line, got {comma_count} in '{line}'"
            );
        }
    }

    #[test]
    fn test_signal_to_svg_sparkline_valid() {
        let signal = Array1::from_vec((0..20).map(|i| (i as f32 * 0.3).sin()).collect());
        let svg = signal_to_svg_sparkline(&signal, 200, 50);
        assert!(svg.contains("<svg"), "missing <svg tag");
        assert!(svg.contains("</svg>"), "missing </svg> tag");
        assert!(svg.contains("polyline"), "missing polyline element");
    }
}
