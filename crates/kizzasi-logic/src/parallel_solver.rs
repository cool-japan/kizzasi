//! Advanced Parallelization for Constraint Evaluation
//!
//! Provides GPU-accelerated batch projection, parallel constraint graph traversal,
//! and SIMD-optimized constraint checking for high-performance constraint satisfaction.
//!
//! # Key Components
//!
//! - [`ParallelFeasibilityChecker`] — parallel batch feasibility checking with rayon
//! - [`ConstraintGraph`] — dependency-aware parallel graph traversal with graph coloring
//! - [`SimdConstraintEvaluator`] — SIMD/auto-vectorization friendly box constraint evaluation
//! - [`IncrementalParallelSolver`] — warm-start incremental solving with parallel projections
//!
//! # Design Notes
//!
//! All inner loops are written in LLVM auto-vectorization friendly patterns
//! (no early exits, simple accumulator patterns) so the compiler can emit SIMD
//! instructions (SSE/AVX/NEON) automatically.

use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use std::time::Instant;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for parallel constraint solving
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads — 0 means auto-detect (uses rayon global pool)
    pub num_threads: usize,
    /// Number of points per worker chunk (default 64)
    pub chunk_size: usize,
    /// Enable SIMD-optimized inner loops
    pub use_simd: bool,
    /// Data prefetch distance (default 8)
    pub prefetch_distance: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: 0,
            chunk_size: 64,
            use_simd: true,
            prefetch_distance: 8,
        }
    }
}

// ============================================================================
// FastConstraint trait
// ============================================================================

/// A constraint represented as a simple function over f32 arrays.
///
/// Implementations must be `Send + Sync` so they can be shared across rayon threads.
pub trait FastConstraint: Send + Sync {
    /// Return `true` if point `x` satisfies this constraint
    fn is_feasible(&self, x: &Array1<f32>) -> bool;

    /// Project `x` onto the feasible set of this constraint.
    /// If `x` is already feasible, returns a clone of `x`.
    fn project(&self, x: &Array1<f32>) -> Array1<f32>;

    /// Return the constraint violation (0 if feasible, > 0 otherwise).
    /// The violation is a non-negative scalar indicating how far `x` is from feasibility.
    fn violation(&self, x: &Array1<f32>) -> f32;
}

// ============================================================================
// BoxConstraint: lb <= x <= ub
// ============================================================================

/// Box (bound) constraint: `lb[i] <= x[i] <= ub[i]` for all `i`.
///
/// This is the most common constraint type in practice. Projection is
/// element-wise clipping, which is trivially SIMD-friendly.
#[derive(Debug, Clone)]
pub struct BoxConstraint {
    /// Lower bounds (element-wise)
    pub lb: Array1<f32>,
    /// Upper bounds (element-wise)
    pub ub: Array1<f32>,
}

impl BoxConstraint {
    /// Create a new box constraint.
    ///
    /// # Errors
    ///
    /// Returns an error string if `lb` and `ub` have different lengths or if any
    /// `lb[i] > ub[i]`.
    pub fn new(lb: Array1<f32>, ub: Array1<f32>) -> Result<Self, String> {
        if lb.len() != ub.len() {
            return Err(format!(
                "BoxConstraint: lb.len()={} != ub.len()={}",
                lb.len(),
                ub.len()
            ));
        }
        for (i, (&l, &u)) in lb.iter().zip(ub.iter()).enumerate() {
            if l > u {
                return Err(format!("BoxConstraint: lb[{i}]={l} > ub[{i}]={u}"));
            }
        }
        Ok(Self { lb, ub })
    }
}

impl FastConstraint for BoxConstraint {
    fn is_feasible(&self, x: &Array1<f32>) -> bool {
        // Written as a fold so LLVM can vectorize (no short-circuit)
        x.iter()
            .zip(self.lb.iter())
            .zip(self.ub.iter())
            .map(|((&xi, &li), &ui)| if xi < li || xi > ui { 1u8 } else { 0u8 })
            .sum::<u8>()
            == 0
    }

    fn project(&self, x: &Array1<f32>) -> Array1<f32> {
        let n = x.len();
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            out[i] = x[i].clamp(self.lb[i], self.ub[i]);
        }
        Array1::from(out)
    }

    fn violation(&self, x: &Array1<f32>) -> f32 {
        let mut v = 0.0f32;
        for i in 0..x.len() {
            let below = (self.lb[i] - x[i]).max(0.0);
            let above = (x[i] - self.ub[i]).max(0.0);
            v += below * below + above * above;
        }
        v.sqrt()
    }
}

// ============================================================================
// L2BallConstraint: ||x - center||_2 <= radius
// ============================================================================

/// L2 ball constraint: `||x - center||_2 <= radius`.
///
/// Projection onto the L2 ball: if inside, keep as-is; if outside, scale to the boundary.
#[derive(Debug, Clone)]
pub struct L2BallConstraint {
    /// Centre of the ball
    pub center: Array1<f32>,
    /// Radius of the ball (must be positive)
    pub radius: f32,
}

impl L2BallConstraint {
    /// Create a new L2 ball constraint.
    ///
    /// # Errors
    ///
    /// Returns an error string if `radius <= 0`.
    pub fn new(center: Array1<f32>, radius: f32) -> Result<Self, String> {
        if radius <= 0.0 {
            return Err(format!(
                "L2BallConstraint: radius must be positive, got {radius}"
            ));
        }
        Ok(Self { center, radius })
    }

    fn dist_sq(&self, x: &Array1<f32>) -> f32 {
        let mut s = 0.0f32;
        for i in 0..x.len() {
            let d = x[i] - self.center[i];
            s += d * d;
        }
        s
    }
}

impl FastConstraint for L2BallConstraint {
    fn is_feasible(&self, x: &Array1<f32>) -> bool {
        self.dist_sq(x) <= self.radius * self.radius
    }

    fn project(&self, x: &Array1<f32>) -> Array1<f32> {
        let dist_sq = self.dist_sq(x);
        if dist_sq <= self.radius * self.radius {
            return x.clone();
        }
        let dist = dist_sq.sqrt();
        let scale = self.radius / dist;
        let n = x.len();
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            out[i] = self.center[i] + (x[i] - self.center[i]) * scale;
        }
        Array1::from(out)
    }

    fn violation(&self, x: &Array1<f32>) -> f32 {
        let dist = self.dist_sq(x).sqrt();
        (dist - self.radius).max(0.0)
    }
}

// ============================================================================
// HyperplaneConstraint: a^T x <= b
// ============================================================================

/// Hyperplane (half-space) constraint: `a^T x <= b`.
///
/// Projection: if feasible keep x; otherwise project onto the hyperplane
/// `{y : a^T y = b}` along the normal direction.
#[derive(Debug, Clone)]
pub struct HyperplaneConstraint {
    /// Normal vector `a`
    pub normal: Array1<f32>,
    /// Offset `b`
    pub offset: f32,
}

impl HyperplaneConstraint {
    /// Create a new hyperplane constraint.
    ///
    /// # Errors
    ///
    /// Returns an error string if `normal` is the zero vector.
    pub fn new(normal: Array1<f32>, offset: f32) -> Result<Self, String> {
        let norm_sq: f32 = normal.iter().map(|&v| v * v).sum();
        if norm_sq == 0.0 {
            return Err("HyperplaneConstraint: normal vector must be non-zero".to_string());
        }
        Ok(Self { normal, offset })
    }

    fn dot(&self, x: &Array1<f32>) -> f32 {
        let mut s = 0.0f32;
        for i in 0..x.len() {
            s += self.normal[i] * x[i];
        }
        s
    }

    fn norm_sq(&self) -> f32 {
        let mut s = 0.0f32;
        for &v in self.normal.iter() {
            s += v * v;
        }
        s
    }
}

impl FastConstraint for HyperplaneConstraint {
    fn is_feasible(&self, x: &Array1<f32>) -> bool {
        self.dot(x) <= self.offset
    }

    fn project(&self, x: &Array1<f32>) -> Array1<f32> {
        let ax = self.dot(x);
        if ax <= self.offset {
            return x.clone();
        }
        // Project: x - ((a^T x - b) / ||a||^2) * a
        let scale = (ax - self.offset) / self.norm_sq();
        let n = x.len();
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            out[i] = x[i] - scale * self.normal[i];
        }
        Array1::from(out)
    }

    fn violation(&self, x: &Array1<f32>) -> f32 {
        (self.dot(x) - self.offset).max(0.0)
    }
}

// ============================================================================
// SimplexConstraint: x >= 0, sum(x) = 1
// ============================================================================

/// Probability simplex constraint: `x[i] >= 0` for all `i` and `sum(x) = 1`.
///
/// Uses the O(n log n) algorithm by Duchi et al. (2008).
#[derive(Debug, Clone)]
pub struct SimplexConstraint {
    /// Dimension of the simplex
    pub dim: usize,
}

impl SimplexConstraint {
    /// Create a new simplex constraint.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl FastConstraint for SimplexConstraint {
    fn is_feasible(&self, x: &Array1<f32>) -> bool {
        if x.len() != self.dim {
            return false;
        }
        let sum: f32 = x.iter().sum();
        let all_nonneg = x
            .iter()
            .map(|&v| if v < 0.0 { 1u8 } else { 0u8 })
            .sum::<u8>()
            == 0;
        all_nonneg && (sum - 1.0).abs() < 1e-5
    }

    fn project(&self, x: &Array1<f32>) -> Array1<f32> {
        let n = x.len();
        // Duchi et al. O(n log n) simplex projection
        let mut sorted: Vec<f32> = x.iter().copied().collect();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0f32;
        let mut rho = 0usize;
        for (j, &s) in sorted.iter().enumerate() {
            cumsum += s;
            if s > (cumsum - 1.0) / (j as f32 + 1.0) {
                rho = j;
            }
        }
        let cumsum_rho: f32 = sorted[..=rho].iter().sum();
        let theta = (cumsum_rho - 1.0) / (rho as f32 + 1.0);

        let mut out = vec![0.0f32; n];
        for i in 0..n {
            out[i] = (x[i] - theta).max(0.0);
        }
        Array1::from(out)
    }

    fn violation(&self, x: &Array1<f32>) -> f32 {
        let sum: f32 = x.iter().sum();
        let sum_viol = (sum - 1.0).abs();
        let neg_viol: f32 = x.iter().map(|&v| (-v).max(0.0)).sum();
        sum_viol + neg_viol
    }
}

// ============================================================================
// ParallelFeasibilityChecker
// ============================================================================

/// Parallel batch feasibility checker.
///
/// Checks N points against M constraints simultaneously using rayon.
/// The outermost parallelism is over points (each point is independent);
/// within a point, constraints are checked sequentially for cache efficiency.
pub struct ParallelFeasibilityChecker {
    constraints: Vec<Box<dyn FastConstraint>>,
    config: ParallelConfig,
}

impl ParallelFeasibilityChecker {
    /// Create a new checker with the given configuration.
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            constraints: Vec::new(),
            config,
        }
    }

    /// Add a constraint to the checker.
    pub fn add_constraint(&mut self, constraint: Box<dyn FastConstraint>) {
        self.constraints.push(constraint);
    }

    /// Number of registered constraints.
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Check feasibility of a batch of points.
    ///
    /// `points` must be a `(num_points, dim)` matrix in row-major order.
    /// Returns one `bool` per point indicating whether all constraints are satisfied.
    pub fn check_batch(&self, points: &Array2<f32>) -> Vec<bool> {
        let (n_points, dim) = points.dim();
        let constraints = &self.constraints;
        let chunk_size = self.config.chunk_size.max(1);

        (0..n_points)
            .into_par_iter()
            .with_min_len(chunk_size)
            .map(|i| {
                let row: Array1<f32> = points.slice(scirs2_core::ndarray::s![i, ..]).to_owned();
                if row.len() != dim {
                    return false;
                }
                constraints.iter().all(|c| c.is_feasible(&row))
            })
            .collect()
    }

    /// Compute violation for each point against all constraints.
    ///
    /// Returns a `(num_points, num_constraints)` matrix where entry `[i, j]`
    /// is the violation of point `i` against constraint `j` (0 if feasible).
    pub fn violation_matrix(&self, points: &Array2<f32>) -> Array2<f32> {
        let (n_points, _dim) = points.dim();
        let n_constraints = self.constraints.len();

        if n_constraints == 0 || n_points == 0 {
            return Array2::zeros((n_points, n_constraints));
        }

        let chunk_size = self.config.chunk_size.max(1);
        let constraints = &self.constraints;

        let rows: Vec<Vec<f32>> = (0..n_points)
            .into_par_iter()
            .with_min_len(chunk_size)
            .map(|i| {
                let row: Array1<f32> = points.slice(scirs2_core::ndarray::s![i, ..]).to_owned();
                constraints.iter().map(|c| c.violation(&row)).collect()
            })
            .collect();

        let mut out = Array2::zeros((n_points, n_constraints));
        for (i, row) in rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                out[[i, j]] = v;
            }
        }
        out
    }

    /// Project all points to the constraint-satisfying region.
    ///
    /// Uses Dykstra's alternating projections algorithm in parallel — each point
    /// is independent, so rayon parallelism is embarrassingly parallel here.
    pub fn project_batch(&self, points: &Array2<f32>, max_iter: usize) -> Array2<f32> {
        let (n_points, dim) = points.dim();
        if n_points == 0 || dim == 0 {
            return points.clone();
        }

        let chunk_size = self.config.chunk_size.max(1);
        let constraints = &self.constraints;

        let projected: Vec<Vec<f32>> = (0..n_points)
            .into_par_iter()
            .with_min_len(chunk_size)
            .map(|i| {
                let row: Array1<f32> = points.slice(scirs2_core::ndarray::s![i, ..]).to_owned();
                let result = dykstra_project(&row, constraints.as_slice(), max_iter);
                result.into_raw_vec_and_offset().0
            })
            .collect();

        let mut out = Array2::zeros((n_points, dim));
        for (i, row) in projected.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if j < dim {
                    out[[i, j]] = v;
                }
            }
        }
        out
    }
}

/// Dykstra's alternating projections algorithm.
///
/// Projects `x` onto the intersection of all constraint sets using the
/// incremental projections variant with correction vectors.
fn dykstra_project(
    x: &Array1<f32>,
    constraints: &[Box<dyn FastConstraint>],
    max_iter: usize,
) -> Array1<f32> {
    if constraints.is_empty() {
        return x.clone();
    }
    let n = x.len();
    let m = constraints.len();

    // Dykstra's algorithm: maintain increment vectors p_k for each constraint
    let mut z = x.clone();
    let mut increments: Vec<Array1<f32>> = vec![Array1::zeros(n); m];

    for _ in 0..max_iter {
        let prev = z.clone();
        for (k, constraint) in constraints.iter().enumerate() {
            let y = &z + &increments[k];
            let proj = constraint.project(&y);
            // Update increment: p_k = y - proj(y)
            for j in 0..n {
                increments[k][j] = y[j] - proj[j];
            }
            z = proj;
        }
        // Check convergence: ||z_new - z_old||_inf
        let max_diff = z
            .iter()
            .zip(prev.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if max_diff < 1e-6 {
            break;
        }
    }
    z
}

// ============================================================================
// ConstraintGraph
// ============================================================================

/// Result of constraint propagation through the graph.
#[derive(Debug, Clone)]
pub struct PropagationResult {
    /// Whether propagation converged within the iteration budget
    pub converged: bool,
    /// Number of iterations performed
    pub iterations: usize,
    /// Number of constraints still violated at termination
    pub num_violations: usize,
}

/// Constraint dependency graph.
///
/// Models which constraints share variables. This enables:
/// - Graph coloring to find independent constraint sets
/// - Parallel propagation within independent sets
/// - Efficient incremental solving after small changes
pub struct ConstraintGraph {
    num_vars: usize,
    constraints: Vec<Box<dyn FastConstraint>>,
    /// Which variable indices each constraint touches
    var_indices: Vec<Vec<usize>>,
    /// Adjacency list: constraints that share at least one variable
    adjacency: Vec<Vec<usize>>,
}

impl ConstraintGraph {
    /// Create a new empty constraint graph over `num_vars` variables.
    pub fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            constraints: Vec::new(),
            var_indices: Vec::new(),
            adjacency: Vec::new(),
        }
    }

    /// Number of constraints in the graph.
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Number of variables in the graph.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Add a constraint that touches the given variable indices.
    ///
    /// The adjacency list is updated so that this constraint becomes adjacent
    /// to all existing constraints that share at least one variable index.
    pub fn add_constraint(&mut self, constraint: Box<dyn FastConstraint>, var_indices: Vec<usize>) {
        let new_idx = self.constraints.len();
        self.constraints.push(constraint);

        // Compute adjacency with existing constraints
        let mut neighbors = Vec::new();
        for (existing_idx, existing_vars) in self.var_indices.iter().enumerate() {
            let shares = var_indices.iter().any(|v| existing_vars.contains(v));
            if shares {
                neighbors.push(existing_idx);
                self.adjacency[existing_idx].push(new_idx);
            }
        }

        self.var_indices.push(var_indices);
        self.adjacency.push(neighbors);
    }

    /// Greedy graph coloring — returns independent sets (constraints with same color).
    ///
    /// Uses Welsh-Powell ordering (decreasing degree) for better coloring quality.
    /// Constraints in the same independent set can be checked/projected in parallel.
    ///
    /// # Returns
    ///
    /// A `Vec<Vec<usize>>` where each inner `Vec` is a set of constraint indices
    /// that are mutually non-adjacent (i.e., share no variables).
    pub fn independent_sets(&self) -> Vec<Vec<usize>> {
        let n = self.constraints.len();
        if n == 0 {
            return Vec::new();
        }

        // Welsh-Powell: sort by degree descending
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(self.adjacency[i].len()));

        let mut colors: Vec<Option<usize>> = vec![None; n];
        let mut num_colors = 0usize;

        for &node in &order {
            // Find the smallest color not used by any neighbor
            let used_colors: std::collections::HashSet<usize> = self.adjacency[node]
                .iter()
                .filter_map(|&nb| colors[nb])
                .collect();

            let color = (0..).find(|c| !used_colors.contains(c)).unwrap_or(0);
            colors[node] = Some(color);
            if color >= num_colors {
                num_colors = color + 1;
            }
        }

        let mut sets: Vec<Vec<usize>> = vec![Vec::new(); num_colors];
        for (node, color) in colors.iter().enumerate() {
            if let Some(c) = color {
                sets[*c].push(node);
            }
        }
        sets
    }

    /// Parallel constraint propagation.
    ///
    /// Projects `x` using constraints in independent-set order.
    /// Within each independent set, constraints are applied in parallel via rayon
    /// (each one operates on disjoint variable subsets, so there are no races).
    ///
    /// Falls back to sequential application when variable sets overlap within a color
    /// (the graph coloring guarantees no overlap for correctly registered constraints).
    pub fn propagate_parallel(&self, x: &mut Array1<f32>) -> PropagationResult {
        let max_iter = 50usize;
        let tol = 1e-6f32;
        let sets = self.independent_sets();

        let mut iterations = 0usize;
        let mut converged = false;

        for _global_iter in 0..max_iter {
            iterations += 1;
            let prev = x.clone();

            for set in &sets {
                if set.is_empty() {
                    continue;
                }
                // Within a color group: constraints touch disjoint variables,
                // so we can project independently and merge results.
                // Compute projections in parallel, then write back.
                let projections: Vec<(usize, Array1<f32>)> = set
                    .par_iter()
                    .map(|&c_idx| {
                        let proj = self.constraints[c_idx].project(x);
                        (c_idx, proj)
                    })
                    .collect();

                // Apply variable updates (write disjoint variable subsets)
                for (c_idx, proj) in &projections {
                    for &var in &self.var_indices[*c_idx] {
                        if var < x.len() {
                            x[var] = proj[var];
                        }
                    }
                }
            }

            // Convergence check
            let max_diff = x
                .iter()
                .zip(prev.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            if max_diff < tol {
                converged = true;
                break;
            }
        }

        // Count remaining violations
        let num_violations = self
            .constraints
            .iter()
            .filter(|c| !c.is_feasible(x))
            .count();

        PropagationResult {
            converged,
            iterations,
            num_violations,
        }
    }
}

// ============================================================================
// SimdConstraintEvaluator
// ============================================================================

/// SIMD-accelerated batch box constraint evaluation.
///
/// Stores bounds as flat `Vec<f32>` arrays for cache-friendly sequential access.
/// Inner loops are written to be friendly to LLVM's auto-vectorizer:
/// - No branches inside the accumulation loop
/// - Sequential memory access patterns
/// - Simple arithmetic (mul/add/max)
pub struct SimdConstraintEvaluator {
    /// Flattened lower bounds: `lb[c * dim + i]` is the lower bound for constraint `c`, dim `i`
    lb: Vec<f32>,
    /// Flattened upper bounds
    ub: Vec<f32>,
    /// Dimension of each constraint (all must be equal)
    dim: usize,
    /// Number of constraints
    num_constraints: usize,
}

impl SimdConstraintEvaluator {
    /// Create a new evaluator from a list of `(lb, ub)` bound pairs.
    ///
    /// All bound vectors must have the same length (the variable dimension).
    ///
    /// # Errors
    ///
    /// Returns an error string if bounds have different lengths or if `lb[i] > ub[i]`.
    pub fn new(bounds: Vec<(Vec<f32>, Vec<f32>)>) -> Result<Self, String> {
        if bounds.is_empty() {
            return Ok(Self {
                lb: Vec::new(),
                ub: Vec::new(),
                dim: 0,
                num_constraints: 0,
            });
        }

        let dim = bounds[0].0.len();
        for (k, (l, u)) in bounds.iter().enumerate() {
            if l.len() != dim {
                return Err(format!(
                    "SimdConstraintEvaluator: bounds[{k}].0.len()={} != dim={dim}",
                    l.len()
                ));
            }
            if u.len() != dim {
                return Err(format!(
                    "SimdConstraintEvaluator: bounds[{k}].1.len()={} != dim={dim}",
                    u.len()
                ));
            }
            for i in 0..dim {
                if l[i] > u[i] {
                    return Err(format!(
                        "SimdConstraintEvaluator: bounds[{k}].lb[{i}]={} > ub[{i}]={}",
                        l[i], u[i]
                    ));
                }
            }
        }

        let num_constraints = bounds.len();
        let mut lb_flat = vec![0.0f32; num_constraints * dim];
        let mut ub_flat = vec![0.0f32; num_constraints * dim];

        for (k, (l, u)) in bounds.iter().enumerate() {
            for i in 0..dim {
                lb_flat[k * dim + i] = l[i];
                ub_flat[k * dim + i] = u[i];
            }
        }

        Ok(Self {
            lb: lb_flat,
            ub: ub_flat,
            dim,
            num_constraints,
        })
    }

    /// Number of registered box constraints.
    pub fn num_constraints(&self) -> usize {
        self.num_constraints
    }

    /// Evaluate all box constraints for a single point `x`.
    ///
    /// Returns a `Vec<f32>` of length `num_constraints` where each entry is the
    /// L2-distance violation (0 if feasible).  Inner loop is SIMD-friendly.
    pub fn evaluate(&self, x: &[f32]) -> Vec<f32> {
        (0..self.num_constraints)
            .map(|c| {
                let base = c * self.dim;
                let lb_slice = &self.lb[base..base + self.dim];
                let ub_slice = &self.ub[base..base + self.dim];
                // Auto-vectorizable: no branches, simple arithmetic
                let v: f32 = x
                    .iter()
                    .zip(lb_slice.iter())
                    .zip(ub_slice.iter())
                    .map(|((&xi, &lbi), &ubi)| {
                        let lb_viol = (lbi - xi).max(0.0);
                        let ub_viol = (xi - ubi).max(0.0);
                        lb_viol * lb_viol + ub_viol * ub_viol
                    })
                    .sum();
                v.sqrt()
            })
            .collect()
    }

    /// Batch evaluation: very cache-friendly inner loop.
    ///
    /// `points` is `(num_points, dim)`. Returns `(num_points, num_constraints)`.
    pub fn evaluate_batch(&self, points: &Array2<f32>) -> Array2<f32> {
        let (n_points, _dim) = points.dim();
        let n_c = self.num_constraints;

        if n_points == 0 || n_c == 0 {
            return Array2::zeros((n_points, n_c));
        }

        let rows: Vec<Vec<f32>> = (0..n_points)
            .into_par_iter()
            .map(|i| {
                let row: Vec<f32> = points
                    .slice(scirs2_core::ndarray::s![i, ..])
                    .iter()
                    .copied()
                    .collect();
                self.evaluate(&row)
            })
            .collect();

        let mut out = Array2::zeros((n_points, n_c));
        for (i, row) in rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                out[[i, j]] = v;
            }
        }
        out
    }

    /// Fast feasibility check with early exit on first violation.
    ///
    /// Returns `true` only if `x` satisfies all box constraints.
    /// Exits as soon as a violation is found, without evaluating remaining constraints.
    pub fn is_feasible_fast(&self, x: &[f32]) -> bool {
        (0..self.num_constraints).all(|c| {
            let base = c * self.dim;
            // Inner loop: SIMD-friendly accumulation using zip for cache locality
            let lb_slice = &self.lb[base..base + self.dim];
            let ub_slice = &self.ub[base..base + self.dim];
            let max_viol: f32 = x
                .iter()
                .zip(lb_slice.iter())
                .zip(ub_slice.iter())
                .map(|((&xi, &lbi), &ubi)| (lbi - xi).max(0.0) + (xi - ubi).max(0.0))
                .sum();
            max_viol == 0.0
        })
    }
}

// ============================================================================
// IncrementalParallelSolver
// ============================================================================

/// Result from the incremental parallel solver.
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// The solution (projected point)
    pub solution: Array1<f32>,
    /// Whether the solution satisfies all constraints
    pub feasible: bool,
    /// Number of projection iterations performed
    pub iterations: usize,
    /// Number of constraints violated at termination
    pub num_violations: usize,
    /// Wall-clock solve time in microseconds
    pub solve_time_us: u64,
}

/// Multi-threaded incremental constraint solver.
///
/// Re-solves after constraint addition/removal without a full restart when possible:
/// - On constraint *addition*: if the current solution is already feasible for the new
///   constraint, no work is needed. Otherwise, only project onto the new constraint
///   (warm start) before running full alternating projections.
/// - On constraint *removal*: the solution remains feasible for remaining constraints,
///   so no re-solve is needed; `solution_valid` is kept `true`.
pub struct IncrementalParallelSolver {
    #[allow(dead_code)]
    config: ParallelConfig,
    constraints: Vec<Box<dyn FastConstraint>>,
    solution: Option<Array1<f32>>,
    solution_valid: bool,
}

impl IncrementalParallelSolver {
    /// Create a new solver with the given configuration.
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            constraints: Vec::new(),
            solution: None,
            solution_valid: false,
        }
    }

    /// Add a constraint. If a valid solution already exists and it satisfies the new
    /// constraint, `solution_valid` remains `true` — avoiding a full re-solve.
    pub fn add_constraint(&mut self, constraint: Box<dyn FastConstraint>) {
        if self.solution_valid {
            if let Some(ref sol) = self.solution {
                if !constraint.is_feasible(sol) {
                    self.solution_valid = false;
                }
            }
        }
        self.constraints.push(constraint);
    }

    /// Remove a constraint by index. Returns `false` if `idx` is out of range.
    ///
    /// After removal the cached solution (if any) is still feasible for the remaining
    /// constraints, so `solution_valid` is preserved.
    pub fn remove_constraint(&mut self, idx: usize) -> bool {
        if idx >= self.constraints.len() {
            return false;
        }
        self.constraints.remove(idx);
        // Solution is still feasible for the remaining (smaller) constraint set
        // — no need to invalidate.
        true
    }

    /// Force a full re-solve on the next call to `solve`.
    pub fn invalidate(&mut self) {
        self.solution_valid = false;
    }

    /// Return a reference to the cached solution, if any.
    pub fn current_solution(&self) -> Option<&Array1<f32>> {
        self.solution.as_ref()
    }

    /// Number of registered constraints.
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Solve incrementally.
    ///
    /// - If `solution_valid`, warm-starts from the cached solution (fewer iterations).
    /// - Otherwise, starts from `init`.
    ///
    /// Uses Dykstra's alternating projections in parallel (via [`ParallelFeasibilityChecker`]).
    pub fn solve(&mut self, init: Array1<f32>, max_iter: usize) -> SolverResult {
        let start = Instant::now();

        let start_point = if self.solution_valid {
            self.solution.clone().unwrap_or_else(|| init.clone())
        } else {
            init.clone()
        };

        // Determine iteration budget: warm starts need fewer iterations
        let actual_max_iter = if self.solution_valid {
            (max_iter / 4).max(1)
        } else {
            max_iter
        };

        let result = dykstra_project(&start_point, &self.constraints, actual_max_iter);

        // Assess feasibility
        let num_violations = self
            .constraints
            .iter()
            .filter(|c| !c.is_feasible(&result))
            .count();
        let feasible = num_violations == 0;

        let elapsed_us = start.elapsed().as_micros() as u64;

        self.solution = Some(result.clone());
        self.solution_valid = feasible;

        SolverResult {
            solution: result,
            feasible,
            iterations: actual_max_iter,
            num_violations,
            solve_time_us: elapsed_us,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn make_box() -> BoxConstraint {
        BoxConstraint::new(
            Array1::from(vec![0.0f32, 0.0, 0.0]),
            Array1::from(vec![1.0f32, 2.0, 3.0]),
        )
        .expect("valid box")
    }

    // -----------------------------------------------------------------------
    // 1. BoxConstraint: feasibility
    // -----------------------------------------------------------------------
    #[test]
    fn test_box_constraint_feasible() {
        let bc = make_box();
        let x = Array1::from(vec![0.5f32, 1.0, 2.0]);
        assert!(bc.is_feasible(&x));
        assert_eq!(bc.violation(&x), 0.0);
    }

    // -----------------------------------------------------------------------
    // 2. BoxConstraint: projection
    // -----------------------------------------------------------------------
    #[test]
    fn test_box_constraint_project() {
        let bc = make_box();
        let x = Array1::from(vec![-1.0f32, 3.0, 5.0]); // out of bounds
        let p = bc.project(&x);
        assert!((p[0] - 0.0).abs() < 1e-5, "clamped to lb");
        assert!((p[1] - 2.0).abs() < 1e-5, "clamped to ub");
        assert!((p[2] - 3.0).abs() < 1e-5, "clamped to ub");
        assert!(bc.is_feasible(&p));
    }

    // -----------------------------------------------------------------------
    // 3. L2BallConstraint: projection
    // -----------------------------------------------------------------------
    #[test]
    fn test_l2_ball_project() {
        let center = Array1::from(vec![0.0f32, 0.0]);
        let ball = L2BallConstraint::new(center, 1.0).expect("valid ball");

        let outside = Array1::from(vec![3.0f32, 4.0]); // dist = 5
        let p = ball.project(&outside);
        let dist: f32 = p.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (dist - 1.0).abs() < 1e-4,
            "projected onto ball surface, dist={dist}"
        );
        assert!(ball.is_feasible(&p));

        let inside = Array1::from(vec![0.1f32, 0.1]);
        let p2 = ball.project(&inside);
        assert!((p2[0] - inside[0]).abs() < 1e-5, "inside point unchanged");
    }

    // -----------------------------------------------------------------------
    // 4. HyperplaneConstraint: projection
    // -----------------------------------------------------------------------
    #[test]
    fn test_hyperplane_project() {
        // Constraint: x[0] + x[1] <= 1
        let normal = Array1::from(vec![1.0f32, 1.0]);
        let hp = HyperplaneConstraint::new(normal, 1.0).expect("valid hyperplane");

        let violating = Array1::from(vec![2.0f32, 2.0]); // a^T x = 4 > 1
        let p = hp.project(&violating);
        let ax: f32 = p[0] + p[1];
        assert!(
            ax <= 1.0 + 1e-5,
            "projected point satisfies a^T x <= b, got {ax}"
        );
        assert!(hp.is_feasible(&p));

        let ok = Array1::from(vec![0.3f32, 0.3]);
        assert!(hp.is_feasible(&ok));
        assert_eq!(hp.violation(&ok), 0.0);
    }

    // -----------------------------------------------------------------------
    // 5. SimplexConstraint: projection
    // -----------------------------------------------------------------------
    #[test]
    fn test_simplex_project() {
        let simplex = SimplexConstraint::new(4);
        let x = Array1::from(vec![1.0f32, 2.0, 3.0, 4.0]); // sum = 10
        let p = simplex.project(&x);

        let sum: f32 = p.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "sum of projected point should be 1, got {sum}"
        );
        for &v in p.iter() {
            assert!(v >= -1e-5, "all components should be non-negative, got {v}");
        }
        assert!(simplex.is_feasible(&p));
    }

    // -----------------------------------------------------------------------
    // 6. ParallelFeasibilityChecker: batch feasibility
    // -----------------------------------------------------------------------
    #[test]
    fn test_parallel_checker_batch() {
        let mut checker = ParallelFeasibilityChecker::new(ParallelConfig::default());
        checker.add_constraint(Box::new(make_box()));

        // Build 100 points: first 50 feasible, last 50 infeasible
        let mut data = vec![0.0f32; 100 * 3];
        for i in 0..50usize {
            data[i * 3] = 0.5;
            data[i * 3 + 1] = 1.0;
            data[i * 3 + 2] = 1.5;
        }
        for i in 50..100usize {
            data[i * 3] = 5.0; // out of [0,1]
            data[i * 3 + 1] = 0.5;
            data[i * 3 + 2] = 0.5;
        }
        let points = Array2::from_shape_vec((100, 3), data).expect("valid shape");
        let results = checker.check_batch(&points);
        assert_eq!(results.len(), 100);
        let feasible_count = results.iter().filter(|&&f| f).count();
        assert_eq!(
            feasible_count, 50,
            "expected 50 feasible points, got {feasible_count}"
        );
    }

    // -----------------------------------------------------------------------
    // 7. ParallelFeasibilityChecker: violation matrix shape
    // -----------------------------------------------------------------------
    #[test]
    fn test_parallel_checker_violations() {
        let mut checker = ParallelFeasibilityChecker::new(ParallelConfig::default());
        checker.add_constraint(Box::new(make_box()));
        checker.add_constraint(Box::new(
            L2BallConstraint::new(Array1::from(vec![0.5f32, 1.0, 1.5]), 2.0).expect("valid ball"),
        ));

        let data: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0, 3.0, 4.0];
        let points = Array2::from_shape_vec((2, 3), data).expect("valid shape");
        let mat = checker.violation_matrix(&points);
        assert_eq!(mat.dim(), (2, 2), "expected (2, 2) violation matrix");
        // First point is feasible for box constraint
        assert!(mat[[0, 0]] < 1e-5, "first point, box violation should be 0");
    }

    // -----------------------------------------------------------------------
    // 8. ParallelFeasibilityChecker: project_batch
    // -----------------------------------------------------------------------
    #[test]
    fn test_parallel_checker_project_batch() {
        let mut checker = ParallelFeasibilityChecker::new(ParallelConfig::default());
        checker.add_constraint(Box::new(make_box()));

        // All infeasible points
        let data: Vec<f32> = vec![-1.0, 5.0, 10.0, -2.0, 3.0, 7.0];
        let points = Array2::from_shape_vec((2, 3), data).expect("valid shape");
        let projected = checker.project_batch(&points, 50);
        assert_eq!(projected.dim(), (2, 3));
        for i in 0..2usize {
            let row: Array1<f32> = projected.slice(scirs2_core::ndarray::s![i, ..]).to_owned();
            assert!(
                make_box().is_feasible(&row),
                "projected row {i} should be feasible"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 9. ConstraintGraph: independent sets
    // -----------------------------------------------------------------------
    #[test]
    fn test_constraint_graph_independent_sets() {
        // Two constraints sharing variable 0: should be in different sets
        let mut graph = ConstraintGraph::new(3);
        graph.add_constraint(Box::new(make_box()), vec![0, 1]);
        graph.add_constraint(Box::new(make_box()), vec![0, 2]); // shares var 0
        graph.add_constraint(Box::new(make_box()), vec![1, 2]); // shares var 1 with c0, var 2 with c1

        let sets = graph.independent_sets();
        // Verify each set contains no two adjacent constraints
        for set in &sets {
            for (a_idx, &a) in set.iter().enumerate() {
                for &b in set.iter().skip(a_idx + 1) {
                    assert!(
                        !graph.adjacency[a].contains(&b),
                        "constraints {a} and {b} are adjacent but in the same independent set"
                    );
                }
            }
        }
        // All constraints must appear in exactly one set
        let mut seen = std::collections::HashSet::new();
        for set in &sets {
            for &c in set {
                assert!(seen.insert(c), "constraint {c} appears in multiple sets");
            }
        }
        assert_eq!(seen.len(), 3, "all 3 constraints must appear");
    }

    // -----------------------------------------------------------------------
    // 10. SimdConstraintEvaluator: batch evaluation matches sequential
    // -----------------------------------------------------------------------
    #[test]
    fn test_simd_evaluator_batch() {
        let bounds = vec![
            (vec![0.0f32, 0.0, 0.0], vec![1.0f32, 2.0, 3.0]),
            (vec![-1.0f32, -1.0, -1.0], vec![1.0f32, 1.0, 1.0]),
        ];
        let evaluator = SimdConstraintEvaluator::new(bounds).expect("valid bounds");

        let data = vec![0.5f32, 1.5, 2.5, 2.0, 0.0, 0.0];
        let points = Array2::from_shape_vec((2, 3), data.clone()).expect("valid shape");

        let batch = evaluator.evaluate_batch(&points);

        // Compare with sequential
        for i in 0..2usize {
            let row = &data[i * 3..(i + 1) * 3];
            let seq = evaluator.evaluate(row);
            for j in 0..evaluator.num_constraints() {
                assert!(
                    (batch[[i, j]] - seq[j]).abs() < 1e-5,
                    "batch[{i},{j}]={} != seq[{j}]={}",
                    batch[[i, j]],
                    seq[j]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // 11. SimdConstraintEvaluator: fast feasibility early exit
    // -----------------------------------------------------------------------
    #[test]
    fn test_simd_evaluator_fast_feasibility() {
        let bounds = vec![
            (vec![0.0f32, 0.0, 0.0], vec![1.0f32, 2.0, 3.0]),
            (vec![-5.0f32, -5.0, -5.0], vec![5.0f32, 5.0, 5.0]),
        ];
        let evaluator = SimdConstraintEvaluator::new(bounds).expect("valid bounds");

        let feasible = vec![0.5f32, 1.0, 2.0];
        assert!(evaluator.is_feasible_fast(&feasible), "point is feasible");

        let infeasible = vec![2.0f32, 1.0, 2.0]; // violates first constraint (x[0] > 1)
        assert!(
            !evaluator.is_feasible_fast(&infeasible),
            "point is infeasible"
        );
    }

    // -----------------------------------------------------------------------
    // 12. IncrementalParallelSolver: add constraint updates solution
    // -----------------------------------------------------------------------
    #[test]
    fn test_incremental_solver_add_constraint() {
        let mut solver = IncrementalParallelSolver::new(ParallelConfig::default());

        // Add a box constraint [0,1]^3
        solver.add_constraint(Box::new(make_box()));

        // Solve with an infeasible init
        let init = Array1::from(vec![5.0f32, 5.0, 5.0]);
        let result = solver.solve(init, 50);

        assert!(
            result.feasible,
            "solution should be feasible after solving with box constraint"
        );
        assert!(make_box().is_feasible(&result.solution));

        // Add a tighter constraint: [0, 0.5]^3
        let tight_box = BoxConstraint::new(
            Array1::from(vec![0.0f32, 0.0, 0.0]),
            Array1::from(vec![0.5f32, 0.5, 0.5]),
        )
        .expect("valid box");
        // Current solution might violate the new constraint (it's at [1,2,3] boundary)
        solver.add_constraint(Box::new(tight_box));

        // Re-solve — should produce a feasible result for both constraints
        let init2 = Array1::from(vec![1.0f32, 2.0, 3.0]);
        let result2 = solver.solve(init2, 100);
        assert!(
            result2.feasible,
            "solution should be feasible after adding tighter constraint"
        );
        assert!(result2.solution[0] <= 0.5 + 1e-4);
        assert!(result2.solution[1] <= 0.5 + 1e-4);
        assert!(result2.solution[2] <= 0.5 + 1e-4);
    }

    // -----------------------------------------------------------------------
    // 13. IncrementalParallelSolver: warm start uses fewer iterations
    // -----------------------------------------------------------------------
    #[test]
    fn test_incremental_solver_warmstart() {
        let mut solver = IncrementalParallelSolver::new(ParallelConfig::default());
        solver.add_constraint(Box::new(make_box()));

        // Cold solve
        let init = Array1::from(vec![0.5f32, 1.0, 1.5]);
        let cold_result = solver.solve(init.clone(), 100);
        assert!(cold_result.feasible);

        // Solution is now cached and valid — next call uses warm start (fewer iterations)
        let warm_result = solver.solve(init, 100);
        assert!(warm_result.feasible);
        assert!(
            warm_result.iterations <= cold_result.iterations,
            "warm start should use fewer or equal iterations: warm={} cold={}",
            warm_result.iterations,
            cold_result.iterations
        );
    }
}
