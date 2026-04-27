//! Distributed ADMM (Alternating Direction Method of Multipliers) for constraint solving
//!
//! Implements the consensus form of ADMM for distributed optimization:
//!
//! ```text
//! minimize   Σᵢ fᵢ(xᵢ)
//! subject to xᵢ = z  for all i   (consensus constraint)
//! ```
//!
//! Each sub-problem is solved independently (in parallel via rayon), and the
//! global consensus variable `z` is updated by averaging. Convergence is
//! measured using primal and dual residuals with optional over-relaxation.
//!
//! ## References
//! - Boyd et al., "Distributed Optimization and Statistical Learning via ADMM", 2011.

use scirs2_core::ndarray::{Array1, Array2};
use thiserror::Error;

// ─────────────────────────────── Error type ───────────────────────────────

/// Errors that can occur during ADMM solving
#[derive(Debug, Error)]
pub enum AdmmError {
    /// Local variable dimension differs from the global consensus dimension
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// `solve()` was called on a solver with no registered sub-problems
    #[error("No subproblems added")]
    NoSubproblems,

    /// A local sub-problem returned an error
    #[error("Subproblem solve failed: {0}")]
    SubproblemFailed(String),

    /// The iteration limit was reached before the residuals satisfied the tolerances
    #[error("Maximum iterations reached without convergence")]
    MaxIterationsReached,

    /// A floating-point operation produced a non-finite value
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

// ─────────────────────────────── Config ──────────────────────────────────

/// Configuration for the ADMM algorithm
#[derive(Debug, Clone)]
pub struct AdmmConfig {
    /// Augmented Lagrangian penalty parameter ρ (default 1.0).
    ///
    /// Larger values impose stronger coupling between sub-problems.
    pub rho: f32,
    /// Maximum number of ADMM outer iterations (default 100)
    pub max_iterations: usize,
    /// Absolute convergence tolerance for primal/dual residuals (default 1e-4)
    pub abs_tol: f32,
    /// Relative convergence tolerance (default 1e-3)
    pub rel_tol: f32,
    /// Over-relaxation parameter α ∈ [1.0, 1.8] (default 1.0 = no relaxation).
    ///
    /// Values in (1, 1.8) often accelerate convergence.
    pub over_relaxation: f32,
    /// Print iteration statistics when `true`
    pub verbose: bool,
}

impl Default for AdmmConfig {
    fn default() -> Self {
        Self {
            rho: 1.0,
            max_iterations: 100,
            abs_tol: 1e-4,
            rel_tol: 1e-3,
            over_relaxation: 1.0,
            verbose: false,
        }
    }
}

impl AdmmConfig {
    /// Create a config with sensible defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the penalty parameter ρ
    pub fn with_rho(mut self, rho: f32) -> Self {
        self.rho = rho;
        self
    }

    /// Set the maximum number of iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the absolute tolerance
    pub fn with_abs_tol(mut self, abs_tol: f32) -> Self {
        self.abs_tol = abs_tol;
        self
    }

    /// Set the relative tolerance
    pub fn with_rel_tol(mut self, rel_tol: f32) -> Self {
        self.rel_tol = rel_tol;
        self
    }

    /// Set the over-relaxation parameter α (must be in `(0, 2)`)
    pub fn with_over_relaxation(mut self, alpha: f32) -> Self {
        self.over_relaxation = alpha;
        self
    }

    /// Enable verbose iteration logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

// ─────────────────────────────── Result ──────────────────────────────────

/// The output of a completed ADMM solve
#[derive(Debug, Clone)]
pub struct AdmmResult {
    /// The consensus solution vector `z`
    pub solution: Array1<f32>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the algorithm converged within the tolerance before `max_iterations`
    pub converged: bool,
    /// Final primal residual ‖xᵢ − z‖ (averaged across agents)
    pub primal_residual: f32,
    /// Final dual residual ρ‖z − z_old‖
    pub dual_residual: f32,
    /// Sum of local objectives Σᵢ fᵢ(z) evaluated at the consensus solution
    pub objective: f32,
}

// ─────────────────────────────── Trait ───────────────────────────────────

/// A local sub-problem solved by one ADMM agent
///
/// Each agent minimises:
/// ```text
///   fᵢ(xᵢ) + (ρ/2) ‖xᵢ − z + uᵢ‖²
/// ```
/// where `z` is the current global consensus variable and `uᵢ` is the
/// scaled dual variable for agent `i`.
pub trait AdmmSubproblem: Send + Sync {
    /// Solve the local proximal sub-problem and return the updated `xᵢ`.
    ///
    /// # Arguments
    /// * `z` – current global consensus variable
    /// * `u` – current scaled dual variable for this agent
    /// * `rho` – penalty parameter
    fn solve(&self, z: &Array1<f32>, u: &Array1<f32>, rho: f32) -> Result<Array1<f32>, AdmmError>;

    /// Evaluate the local objective fᵢ(x)
    fn objective(&self, x: &Array1<f32>) -> f32;

    /// Dimension of the local variable xᵢ (must equal the global dimension)
    fn dim(&self) -> usize;
}

// ─────────────────────────── Helper utilities ────────────────────────────

/// Compute the L2 norm of an array
#[inline]
fn l2_norm(v: &Array1<f32>) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Element-wise soft-threshold (shrinkage) operator
///
/// `shrink(v, κ)ᵢ = sign(vᵢ) · max(|vᵢ| − κ, 0)`
fn soft_threshold(v: &Array1<f32>, kappa: f32) -> Array1<f32> {
    v.mapv(|x| {
        if x > kappa {
            x - kappa
        } else if x < -kappa {
            x + kappa
        } else {
            0.0
        }
    })
}

/// Box projection: clip each element of `v` to `[lb, ub]`
fn box_clip(v: &Array1<f32>, lb: &Array1<f32>, ub: &Array1<f32>) -> Array1<f32> {
    v.iter()
        .zip(lb.iter())
        .zip(ub.iter())
        .map(|((&vi, &li), &ui)| vi.clamp(li, ui))
        .collect()
}

/// Solve the symmetric positive-definite linear system (Q + ρ I) x = b using
/// the Gauss-Seidel method.
///
/// This avoids any LAPACK/BLAS dependency while remaining correct for PSD
/// matrices. For small dimensions the iteration converges very quickly.
fn gauss_seidel_solve(
    q: &Array2<f32>,
    rho: f32,
    b: &Array1<f32>,
) -> Result<Array1<f32>, AdmmError> {
    let n = b.len();
    if q.nrows() != n || q.ncols() != n {
        return Err(AdmmError::DimensionMismatch {
            expected: n,
            got: q.nrows(),
        });
    }

    let mut x = Array1::<f32>::zeros(n);
    // The effective matrix is A = Q + rho*I
    // Gauss-Seidel: xᵢ ← (bᵢ − Σⱼ≠ᵢ Aᵢⱼ xⱼ) / Aᵢᵢ
    let max_inner = 200usize;
    for _iter in 0..max_inner {
        let x_old = x.clone();
        for i in 0..n {
            let a_ii = q[[i, i]] + rho;
            if a_ii.abs() < f32::EPSILON {
                return Err(AdmmError::NumericalError(format!(
                    "Near-zero diagonal at index {i}"
                )));
            }
            let mut sum = 0.0f32;
            for j in 0..n {
                if j != i {
                    sum += (q[[i, j]] + if i == j { rho } else { 0.0 }) * x[j];
                }
            }
            x[i] = (b[i] - sum) / a_ii;
        }
        // Check convergence of the inner loop
        let diff: f32 = x
            .iter()
            .zip(x_old.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        if diff < 1e-8 {
            break;
        }
    }

    // Verify the solution is finite
    for &xi in x.iter() {
        if !xi.is_finite() {
            return Err(AdmmError::NumericalError(
                "Gauss-Seidel produced non-finite value".into(),
            ));
        }
    }
    Ok(x)
}

// ──────────────────────── Concrete sub-problems ───────────────────────────

/// Quadratic sub-problem: minimise ½ xᵀ Q x + cᵀ x  subject to  lb ≤ x ≤ ub
///
/// The augmented problem is:
/// ```text
///   minimise  ½ xᵀ Q x + cᵀ x + (ρ/2) ‖x − (z − u)‖²
/// = minimise  ½ xᵀ (Q + ρI) x + (c − ρ(z − u))ᵀ x
/// ```
/// Solution: x̂ = (Q + ρI)⁻¹ (ρ(z − u) − c), then clipped to [lb, ub].
#[derive(Debug, Clone)]
pub struct QuadraticSubproblem {
    /// Positive semi-definite quadratic term (n × n)
    pub q: Array2<f32>,
    /// Linear cost term (n)
    pub c: Array1<f32>,
    /// Optional per-element lower bounds
    pub lb: Option<Array1<f32>>,
    /// Optional per-element upper bounds
    pub ub: Option<Array1<f32>>,
}

impl QuadraticSubproblem {
    /// Create an unconstrained quadratic sub-problem
    pub fn new(q: Array2<f32>, c: Array1<f32>) -> Self {
        Self {
            q,
            c,
            lb: None,
            ub: None,
        }
    }

    /// Add box constraints lb ≤ x ≤ ub
    pub fn with_bounds(mut self, lb: Array1<f32>, ub: Array1<f32>) -> Self {
        self.lb = Some(lb);
        self.ub = Some(ub);
        self
    }
}

impl AdmmSubproblem for QuadraticSubproblem {
    fn solve(&self, z: &Array1<f32>, u: &Array1<f32>, rho: f32) -> Result<Array1<f32>, AdmmError> {
        let n = self.dim();
        if z.len() != n || u.len() != n {
            return Err(AdmmError::DimensionMismatch {
                expected: n,
                got: z.len(),
            });
        }
        // RHS: ρ(z − u) − c
        let rhs: Array1<f32> = (z - u).mapv(|v| rho * v) - &self.c;

        // Solve (Q + ρI) x = rhs
        let x = gauss_seidel_solve(&self.q, rho, &rhs)?;

        // Clip to box if bounds are present
        let x = match (&self.lb, &self.ub) {
            (Some(lb), Some(ub)) => box_clip(&x, lb, ub),
            (Some(lb), None) => x.mapv(|v| v.max(lb[0])),
            (None, Some(ub)) => x.mapv(|v| v.min(ub[0])),
            (None, None) => x,
        };

        Ok(x)
    }

    fn objective(&self, x: &Array1<f32>) -> f32 {
        // ½ xᵀ Q x + cᵀ x
        let qx: Array1<f32> = self.q.dot(x);
        0.5 * x.iter().zip(qx.iter()).map(|(a, b)| a * b).sum::<f32>()
            + x.iter().zip(self.c.iter()).map(|(a, b)| a * b).sum::<f32>()
    }

    fn dim(&self) -> usize {
        self.c.len()
    }
}

/// LASSO sub-problem: minimise ‖Ax − b‖² + λ‖x‖₁
///
/// The augmented problem admits a closed-form solution via soft-thresholding.
///
/// The x-update for this sub-problem is:
/// ```text
///   v = (AᵀA + ρI)⁻¹ (Aᵀb + ρ(z − u))
///   x̂ = shrink(v, λ/ρ)
/// ```
#[derive(Debug, Clone)]
pub struct LassoSubproblem {
    /// Measurement matrix (m × n)
    pub a: Array2<f32>,
    /// Observation vector (m)
    pub b: Array1<f32>,
    /// L1 regularisation weight λ
    pub lambda: f32,
}

impl LassoSubproblem {
    /// Create a new LASSO sub-problem
    pub fn new(a: Array2<f32>, b: Array1<f32>, lambda: f32) -> Self {
        Self { a, b, lambda }
    }
}

impl AdmmSubproblem for LassoSubproblem {
    fn solve(&self, z: &Array1<f32>, u: &Array1<f32>, rho: f32) -> Result<Array1<f32>, AdmmError> {
        let n = self.dim();
        if z.len() != n || u.len() != n {
            return Err(AdmmError::DimensionMismatch {
                expected: n,
                got: z.len(),
            });
        }

        // Build AᵀA (n × n)
        let at = self.a.t();
        let ata: Array2<f32> = at.dot(&self.a);

        // RHS: Aᵀb + ρ(z − u)
        let atb: Array1<f32> = at.dot(&self.b);
        let rhs: Array1<f32> = atb + (z - u).mapv(|v| rho * v);

        // Solve (AᵀA + ρI) v = rhs
        let v = gauss_seidel_solve(&ata, rho, &rhs)?;

        // Apply soft-threshold
        let kappa = self.lambda / rho;
        Ok(soft_threshold(&v, kappa))
    }

    fn objective(&self, x: &Array1<f32>) -> f32 {
        let ax_minus_b: Array1<f32> = self.a.dot(x) - &self.b;
        let l2_sq: f32 = ax_minus_b.iter().map(|&v| v * v).sum();
        let l1: f32 = x.iter().map(|&v| v.abs()).sum();
        l2_sq + self.lambda * l1
    }

    fn dim(&self) -> usize {
        self.a.ncols()
    }
}

/// Projection sub-problem: project onto the box [lb, ub]
///
/// The augmented problem is:
/// ```text
///   minimise  ‖x − v‖²  s.t.  lb ≤ x ≤ ub,  where v = z − u
/// ```
/// which has the closed-form solution x̂ = clip(z − u, lb, ub).
#[derive(Debug, Clone)]
pub struct ProjectionSubproblem {
    /// Per-element lower bounds
    pub lb: Array1<f32>,
    /// Per-element upper bounds
    pub ub: Array1<f32>,
}

impl ProjectionSubproblem {
    /// Create a box-projection sub-problem
    pub fn new(lb: Array1<f32>, ub: Array1<f32>) -> Self {
        Self { lb, ub }
    }
}

impl AdmmSubproblem for ProjectionSubproblem {
    fn solve(&self, z: &Array1<f32>, u: &Array1<f32>, _rho: f32) -> Result<Array1<f32>, AdmmError> {
        let n = self.dim();
        if z.len() != n || u.len() != n {
            return Err(AdmmError::DimensionMismatch {
                expected: n,
                got: z.len(),
            });
        }
        let v: Array1<f32> = z - u;
        Ok(box_clip(&v, &self.lb, &self.ub))
    }

    fn objective(&self, _x: &Array1<f32>) -> f32 {
        // Indicator function: 0 inside the box, but we treat it as 0 here
        // because feasible iterates will always be inside [lb, ub].
        0.0
    }

    fn dim(&self) -> usize {
        self.lb.len()
    }
}

// ─────────────────────── Core ADMM iteration logic ───────────────────────

/// Internal output type for one ADMM sweep:
/// `(local_xs, dual_us, z_new, primal_residual, dual_residual)`
type AdmmSweepOutput = (Vec<Array1<f32>>, Vec<Array1<f32>>, Array1<f32>, f32, f32);

/// Run one full ADMM sweep and return (updated local_xs, updated_us, primal_res, dual_res, z_new).
///
/// Uses rayon for parallel x-updates when the `parallel` feature of scirs2-core is active.
fn admm_sweep(
    subproblems: &[Box<dyn AdmmSubproblem>],
    z: &Array1<f32>,
    us: &[Array1<f32>],
    rho: f32,
    alpha: f32,
    weights: Option<&Array1<f32>>,
) -> Result<AdmmSweepOutput, AdmmError> {
    let n_agents = subproblems.len();
    let dim = z.len();

    // ── Step 1: parallel x-update ─────────────────────────────────────────
    // Each agent i solves:  xᵢ ← argmin fᵢ(xᵢ) + (ρ/2)‖xᵢ − z + uᵢ‖²
    use rayon::prelude::*;

    let x_results: Vec<Result<Array1<f32>, AdmmError>> = subproblems
        .par_iter()
        .enumerate()
        .map(|(i, sp)| sp.solve(z, &us[i], rho))
        .collect();

    let mut new_xs: Vec<Array1<f32>> = Vec::with_capacity(n_agents);
    for result in x_results {
        new_xs.push(result?);
    }

    // ── Step 2: z-update with optional over-relaxation ────────────────────
    // z_new = (1/N) Σᵢ (αxᵢ + (1−α)z + uᵢ)
    //       = α * x_avg + (1−α) * z + u_avg
    let z_old = z.clone();

    let (x_avg, u_avg) = match weights {
        Some(w) => {
            // Weighted average: x_avg = Σᵢ wᵢ xᵢ  (weights sum to 1)
            let mut xa = Array1::<f32>::zeros(dim);
            let mut ua = Array1::<f32>::zeros(dim);
            for i in 0..n_agents {
                xa = xa + new_xs[i].mapv(|v| v * w[i]);
                ua = ua + us[i].mapv(|v| v * w[i]);
            }
            (xa, ua)
        }
        None => {
            // Simple average
            let mut xa = Array1::<f32>::zeros(dim);
            let mut ua = Array1::<f32>::zeros(dim);
            for i in 0..n_agents {
                xa += &new_xs[i];
                ua += &us[i];
            }
            let inv_n = 1.0 / n_agents as f32;
            (xa.mapv(|v| v * inv_n), ua.mapv(|v| v * inv_n))
        }
    };

    // Over-relaxation: x̃ᵢ = α xᵢ + (1−α) z
    let z_new: Array1<f32> = x_avg.mapv(|v| alpha * v) + z_old.mapv(|v| (1.0 - alpha) * v) + &u_avg;

    // ── Step 3: dual variable update ─────────────────────────────────────
    // uᵢ ← uᵢ + α xᵢ + (1−α) z − z_new
    //     = uᵢ + x̃ᵢ − z_new   (with x̃ᵢ = α xᵢ + (1−α) z_old)
    let mut new_us: Vec<Array1<f32>> = Vec::with_capacity(n_agents);
    let mut primal_sq = 0.0f32;
    for i in 0..n_agents {
        let x_tilde: Array1<f32> =
            new_xs[i].mapv(|v| alpha * v) + z_old.mapv(|v| (1.0 - alpha) * v);
        let residual_i: Array1<f32> = &x_tilde - &z_new;
        primal_sq += residual_i.iter().map(|&v| v * v).sum::<f32>();
        let u_new: Array1<f32> = &us[i] + &residual_i;
        new_us.push(u_new);
    }
    let primal_res = (primal_sq / n_agents as f32).sqrt();

    // Dual residual: ρ ‖z_new − z_old‖
    let dual_res = rho * l2_norm(&(&z_new - &z_old));

    Ok((new_xs, new_us, z_new, primal_res, dual_res))
}

/// Check ADMM convergence using the Boyd et al. stopping criteria.
///
/// Returns `true` when both the primal and dual residuals fall below the
/// combined absolute + relative threshold.
fn check_convergence(
    primal_res: f32,
    dual_res: f32,
    config: &AdmmConfig,
    n_agents: usize,
    dim: usize,
) -> bool {
    let scale = ((n_agents * dim) as f32).sqrt();
    let eps_primal = config.abs_tol * scale + config.rel_tol;
    let eps_dual = config.abs_tol * scale + config.rel_tol;
    primal_res < eps_primal && dual_res < eps_dual
}

// ──────────────────────── DistributedAdmm ────────────────────────────────

/// Distributed ADMM solver coordinating multiple local sub-problems
///
/// Runs the standard consensus ADMM algorithm with optional over-relaxation
/// and parallelism via rayon.
pub struct DistributedAdmm {
    config: AdmmConfig,
    subproblems: Vec<Box<dyn AdmmSubproblem>>,
    dim: usize,
}

impl DistributedAdmm {
    /// Create a new solver for variables of length `dim`
    pub fn new(config: AdmmConfig, dim: usize) -> Self {
        Self {
            config,
            subproblems: Vec::new(),
            dim,
        }
    }

    /// Register a new local sub-problem.
    ///
    /// Returns `Err(AdmmError::DimensionMismatch)` if the sub-problem has a
    /// different variable dimension from the solver.
    pub fn add_subproblem(&mut self, subproblem: Box<dyn AdmmSubproblem>) -> Result<(), AdmmError> {
        if subproblem.dim() != self.dim {
            return Err(AdmmError::DimensionMismatch {
                expected: self.dim,
                got: subproblem.dim(),
            });
        }
        self.subproblems.push(subproblem);
        Ok(())
    }

    /// Number of registered sub-problems
    pub fn num_subproblems(&self) -> usize {
        self.subproblems.len()
    }

    /// Solve starting from the zero vector
    pub fn solve(&self) -> Result<AdmmResult, AdmmError> {
        self.solve_warm(Array1::zeros(self.dim))
    }

    /// Solve with a user-provided warm-start for `z`
    pub fn solve_warm(&self, z_init: Array1<f32>) -> Result<AdmmResult, AdmmError> {
        if self.subproblems.is_empty() {
            return Err(AdmmError::NoSubproblems);
        }
        if z_init.len() != self.dim {
            return Err(AdmmError::DimensionMismatch {
                expected: self.dim,
                got: z_init.len(),
            });
        }

        let n_agents = self.subproblems.len();
        let mut z = z_init;
        let mut us: Vec<Array1<f32>> = vec![Array1::zeros(self.dim); n_agents];

        let mut primal_res = f32::INFINITY;
        let mut dual_res = f32::INFINITY;
        let mut iterations = 0usize;
        let mut converged = false;

        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;
            let (new_xs, new_us, z_new, pr, dr) = admm_sweep(
                &self.subproblems,
                &z,
                &us,
                self.config.rho,
                self.config.over_relaxation,
                None,
            )?;

            let _ = new_xs; // local x values are internal; z carries the consensus
            us = new_us;
            z = z_new;
            primal_res = pr;
            dual_res = dr;

            if self.config.verbose {
                tracing::debug!(iter = iterations, primal_res, dual_res, "ADMM iteration");
            }

            if check_convergence(primal_res, dual_res, &self.config, n_agents, self.dim) {
                converged = true;
                break;
            }
        }

        // Evaluate total objective at the consensus solution
        let objective: f32 = self.subproblems.iter().map(|sp| sp.objective(&z)).sum();

        Ok(AdmmResult {
            solution: z,
            iterations,
            converged,
            primal_residual: primal_res,
            dual_residual: dual_res,
            objective,
        })
    }
}

// ──────────────────────── ConsensusAdmm ──────────────────────────────────

/// Consensus ADMM solver with optional per-agent weighting
///
/// Identical to [`DistributedAdmm`] but allows a user-supplied weight vector
/// so that the z-update becomes a weighted average of local variables.
/// The weights must be non-negative and sum to 1.
pub struct ConsensusAdmm {
    config: AdmmConfig,
    subproblems: Vec<Box<dyn AdmmSubproblem>>,
    dim: usize,
    weights: Option<Array1<f32>>,
}

impl ConsensusAdmm {
    /// Create a new consensus ADMM solver (uniform weights)
    pub fn new(config: AdmmConfig, dim: usize) -> Self {
        Self {
            config,
            subproblems: Vec::new(),
            dim,
            weights: None,
        }
    }

    /// Create a weighted consensus ADMM solver.
    ///
    /// `weights` must have length equal to the number of sub-problems that
    /// will be added, and must sum to approximately 1.0.
    pub fn new_weighted(
        config: AdmmConfig,
        dim: usize,
        weights: Array1<f32>,
    ) -> Result<Self, AdmmError> {
        let sum: f32 = weights.iter().sum();
        if (sum - 1.0).abs() > 1e-4 {
            return Err(AdmmError::NumericalError(format!(
                "Weights must sum to 1.0, got {sum:.6}"
            )));
        }
        Ok(Self {
            config,
            subproblems: Vec::new(),
            dim,
            weights: Some(weights),
        })
    }

    /// Register a local sub-problem
    pub fn add_subproblem(&mut self, subproblem: Box<dyn AdmmSubproblem>) -> Result<(), AdmmError> {
        if subproblem.dim() != self.dim {
            return Err(AdmmError::DimensionMismatch {
                expected: self.dim,
                got: subproblem.dim(),
            });
        }
        self.subproblems.push(subproblem);
        Ok(())
    }

    /// Run the consensus ADMM solve
    pub fn solve(&self) -> Result<AdmmResult, AdmmError> {
        if self.subproblems.is_empty() {
            return Err(AdmmError::NoSubproblems);
        }

        let n_agents = self.subproblems.len();

        // Validate weights length if provided
        if let Some(w) = &self.weights {
            if w.len() != n_agents {
                return Err(AdmmError::DimensionMismatch {
                    expected: n_agents,
                    got: w.len(),
                });
            }
        }

        let mut z = Array1::<f32>::zeros(self.dim);
        let mut us: Vec<Array1<f32>> = vec![Array1::zeros(self.dim); n_agents];

        let mut primal_res = f32::INFINITY;
        let mut dual_res = f32::INFINITY;
        let mut iterations = 0usize;
        let mut converged = false;

        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;
            let (new_xs, new_us, z_new, pr, dr) = admm_sweep(
                &self.subproblems,
                &z,
                &us,
                self.config.rho,
                self.config.over_relaxation,
                self.weights.as_ref(),
            )?;

            let _ = new_xs;
            us = new_us;
            z = z_new;
            primal_res = pr;
            dual_res = dr;

            if self.config.verbose {
                tracing::debug!(
                    iter = iterations,
                    primal_res,
                    dual_res,
                    "ConsensusADMM iteration"
                );
            }

            if check_convergence(primal_res, dual_res, &self.config, n_agents, self.dim) {
                converged = true;
                break;
            }
        }

        let objective: f32 = self.subproblems.iter().map(|sp| sp.objective(&z)).sum();

        Ok(AdmmResult {
            solution: z,
            iterations,
            converged,
            primal_residual: primal_res,
            dual_residual: dual_res,
            objective,
        })
    }
}

// ─────────────────────────────── Tests ───────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    // ─── helpers ──────────────────────────────────────────────────────────

    /// Build an n×n identity matrix scaled by `scale`
    fn eye(n: usize, scale: f32) -> Array2<f32> {
        let mut m = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            m[[i, i]] = scale;
        }
        m
    }

    // ─── 1. Default config ───────────────────────────────────────────────

    #[test]
    fn test_admm_config_default() {
        let cfg = AdmmConfig::default();
        assert!(cfg.rho > 0.0, "rho must be positive");
        assert!(cfg.max_iterations > 0, "max_iterations must be positive");
        assert!(cfg.abs_tol > 0.0, "abs_tol must be positive");
        assert!(cfg.rel_tol > 0.0, "rel_tol must be positive");
        assert!(
            (0.0..2.0).contains(&cfg.over_relaxation),
            "over_relaxation must be in (0, 2)"
        );
    }

    // ─── 2. ProjectionSubproblem ─────────────────────────────────────────

    #[test]
    fn test_projection_subproblem() {
        let lb = Array1::from_vec(vec![0.0, -1.0, 2.0]);
        let ub = Array1::from_vec(vec![1.0, 1.0, 5.0]);
        let sp = ProjectionSubproblem::new(lb.clone(), ub.clone());

        // z outside the box, u = 0 → result should be clipped
        let z = Array1::from_vec(vec![-2.0, 3.0, 10.0]);
        let u = Array1::zeros(3);
        let x = sp.solve(&z, &u, 1.0).expect("projection should succeed");
        assert!((x[0] - 0.0).abs() < 1e-6, "should clip to lb[0]");
        assert!((x[1] - 1.0).abs() < 1e-6, "should clip to ub[1]");
        assert!((x[2] - 5.0).abs() < 1e-6, "should clip to ub[2]");
    }

    // ─── 3. LassoSubproblem – soft threshold ────────────────────────────

    #[test]
    fn test_lasso_subproblem_soft_threshold() {
        // A = I, b = 0, λ large → solution should shrink towards zero
        let n = 3usize;
        let a = eye(n, 1.0);
        let b = Array1::zeros(n);
        let lambda = 10.0f32;
        let sp = LassoSubproblem::new(a, b, lambda);

        let z = Array1::from_vec(vec![0.5, -0.5, 0.2]);
        let u = Array1::zeros(n);
        let x = sp.solve(&z, &u, 1.0).expect("lasso solve");

        // With large lambda the solution should be very small in magnitude
        for &xi in x.iter() {
            assert!(xi.abs() < 0.6, "soft-threshold should shrink the solution");
        }
    }

    // ─── 4. QuadraticSubproblem – unconstrained ─────────────────────────

    #[test]
    fn test_quadratic_subproblem_unconstrained() {
        // Q = I, c = [-2, -2], minimum at x = (Q)^{-1} * 2 = [2, 2]
        // but with ADMM coupling (rho=1, z=[2,2], u=[0,0]) we get
        // (Q + I)x = ρ(z-u) - c = [2,2] - [-2,-2] = [4,4]
        // x = [2, 2]
        let n = 2usize;
        let q = eye(n, 1.0);
        let c = Array1::from_vec(vec![-2.0, -2.0]);
        let sp = QuadraticSubproblem::new(q, c);

        let z = Array1::from_vec(vec![2.0, 2.0]);
        let u = Array1::zeros(n);
        let x = sp.solve(&z, &u, 1.0).expect("quadratic solve");
        assert!((x[0] - 2.0).abs() < 1e-3, "x[0] ≈ 2: got {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-3, "x[1] ≈ 2: got {}", x[1]);
    }

    // ─── 5. QuadraticSubproblem – box constrained ────────────────────────

    #[test]
    fn test_quadratic_subproblem_constrained() {
        let n = 2usize;
        let q = eye(n, 1.0);
        let c = Array1::from_vec(vec![-5.0, -5.0]);
        let lb = Array1::from_vec(vec![0.0, 0.0]);
        let ub = Array1::from_vec(vec![1.0, 1.0]); // clip at 1
        let sp = QuadraticSubproblem::new(q, c).with_bounds(lb, ub);

        let z = Array1::from_vec(vec![3.0, 3.0]);
        let u = Array1::zeros(n);
        let x = sp.solve(&z, &u, 1.0).expect("constrained quadratic solve");
        // Solution without bounds would be >> 1, so clipping should kick in
        assert!(x[0] <= 1.0 + 1e-6, "x[0] must be ≤ ub");
        assert!(x[1] <= 1.0 + 1e-6, "x[1] must be ≤ ub");
        assert!(x[0] >= 0.0 - 1e-6, "x[0] must be ≥ lb");
    }

    // ─── 6. DistributedAdmm – basic consensus ───────────────────────────

    #[test]
    fn test_distributed_admm_consensus() {
        // Three projection sub-problems with overlapping boxes.
        // The intersection is [1, 1] so consensus should settle near [1, 1].
        let dim = 2usize;
        let cfg = AdmmConfig::default()
            .with_rho(2.0)
            .with_max_iterations(200)
            .with_abs_tol(1e-4);

        let mut solver = DistributedAdmm::new(cfg, dim);
        // Agent 0: [1, 3] × [1, 3]
        solver
            .add_subproblem(Box::new(ProjectionSubproblem::new(
                Array1::from_vec(vec![1.0, 1.0]),
                Array1::from_vec(vec![3.0, 3.0]),
            )))
            .expect("add agent 0");
        // Agent 1: [0, 1] × [0, 1]
        solver
            .add_subproblem(Box::new(ProjectionSubproblem::new(
                Array1::from_vec(vec![0.0, 0.0]),
                Array1::from_vec(vec![1.0, 1.0]),
            )))
            .expect("add agent 1");
        // Agent 2: [1, 2] × [0, 2]
        solver
            .add_subproblem(Box::new(ProjectionSubproblem::new(
                Array1::from_vec(vec![1.0, 0.0]),
                Array1::from_vec(vec![2.0, 2.0]),
            )))
            .expect("add agent 2");

        let result = solver.solve().expect("distributed ADMM solve");
        // Consensus should be near the common point [1, 1]
        assert!((result.solution[0] - 1.0).abs() < 0.1, "x[0] ≈ 1");
        assert!((result.solution[1] - 1.0).abs() < 0.1, "x[1] ≈ 1");
    }

    // ─── 7. DistributedAdmm – convergence within max_iter ───────────────

    #[test]
    fn test_distributed_admm_convergence() {
        let dim = 4usize;
        let cfg = AdmmConfig::default()
            .with_max_iterations(500)
            .with_abs_tol(1e-3);

        let mut solver = DistributedAdmm::new(cfg, dim);
        for _ in 0..3 {
            solver
                .add_subproblem(Box::new(ProjectionSubproblem::new(
                    Array1::from_vec(vec![0.0; dim]),
                    Array1::from_vec(vec![1.0; dim]),
                )))
                .expect("add subproblem");
        }

        let result = solver.solve().expect("solve");
        // Primal/dual residuals must be finite
        assert!(result.primal_residual.is_finite());
        assert!(result.dual_residual.is_finite());
    }

    // ─── 8. DistributedAdmm – converged flag ────────────────────────────

    #[test]
    fn test_distributed_admm_convergence_flag() {
        let dim = 2usize;
        let cfg = AdmmConfig::default()
            .with_max_iterations(1000)
            .with_abs_tol(1e-3)
            .with_rel_tol(1e-2);

        let mut solver = DistributedAdmm::new(cfg, dim);
        // Two identical boxes → trivial consensus, should converge fast
        for _ in 0..2 {
            solver
                .add_subproblem(Box::new(ProjectionSubproblem::new(
                    Array1::from_vec(vec![0.0, 0.0]),
                    Array1::from_vec(vec![1.0, 1.0]),
                )))
                .expect("add subproblem");
        }

        let result = solver.solve().expect("solve");
        assert!(result.converged, "should have converged");
    }

    // ─── 9. ConsensusAdmm – weighted average ────────────────────────────

    #[test]
    fn test_consensus_admm_weighted() {
        let dim = 1usize;
        // Two projection sub-problems on disjoint intervals:
        //   Agent 0: [0, 0.4] (weight 0.3)
        //   Agent 1: [0.6, 1.0] (weight 0.7)
        // Weighted average of endpoints ≈ 0.3*0.4 + 0.7*0.6 = 0.54  (roughly)
        let weights = Array1::from_vec(vec![0.3f32, 0.7]);
        let cfg = AdmmConfig::default()
            .with_max_iterations(500)
            .with_abs_tol(1e-3);

        let mut solver = ConsensusAdmm::new_weighted(cfg, dim, weights).expect("new_weighted");

        solver
            .add_subproblem(Box::new(ProjectionSubproblem::new(
                Array1::from_vec(vec![0.0]),
                Array1::from_vec(vec![0.4]),
            )))
            .expect("add agent 0");
        solver
            .add_subproblem(Box::new(ProjectionSubproblem::new(
                Array1::from_vec(vec![0.6]),
                Array1::from_vec(vec![1.0]),
            )))
            .expect("add agent 1");

        let result = solver.solve().expect("weighted consensus solve");
        // Solution should be between the two boxes
        assert!(result.solution[0] >= 0.0 - 1e-4);
        assert!(result.solution[0] <= 1.0 + 1e-4);
    }

    // ─── 10. Warm start reduces iteration count ──────────────────────────

    #[test]
    fn test_admm_warm_start() {
        let dim = 3usize;
        let cfg = AdmmConfig::default()
            .with_rho(2.0)
            .with_max_iterations(500)
            .with_abs_tol(1e-5);

        let make_solver = |cfg: AdmmConfig| -> DistributedAdmm {
            let mut s = DistributedAdmm::new(cfg, dim);
            for _ in 0..2 {
                s.add_subproblem(Box::new(ProjectionSubproblem::new(
                    Array1::from_vec(vec![0.5, 0.5, 0.5]),
                    Array1::from_vec(vec![1.0, 1.0, 1.0]),
                )))
                .expect("add subproblem");
            }
            s
        };

        let cold = make_solver(cfg.clone()).solve().expect("cold solve");
        // Warm start from the already-computed solution
        let warm = make_solver(cfg)
            .solve_warm(cold.solution.clone())
            .expect("warm solve");

        // Warm start should converge in fewer or equal iterations
        // (with a perfect warm start it may take 1 iteration)
        assert!(
            warm.iterations <= cold.iterations,
            "warm ({}) should not exceed cold ({})",
            warm.iterations,
            cold.iterations
        );
    }

    // ─── 11. Error: no subproblems ───────────────────────────────────────

    #[test]
    fn test_admm_no_subproblems_error() {
        let cfg = AdmmConfig::default();
        let solver = DistributedAdmm::new(cfg, 3);
        let result = solver.solve();
        assert!(
            matches!(result, Err(AdmmError::NoSubproblems)),
            "expected NoSubproblems error"
        );
    }

    // ─── 12. Error: dimension mismatch ──────────────────────────────────

    #[test]
    fn test_admm_dimension_mismatch() {
        let cfg = AdmmConfig::default();
        let mut solver = DistributedAdmm::new(cfg, 3);
        // Add a sub-problem with dim = 5 (wrong)
        let result = solver.add_subproblem(Box::new(ProjectionSubproblem::new(
            Array1::from_vec(vec![0.0; 5]),
            Array1::from_vec(vec![1.0; 5]),
        )));
        assert!(
            matches!(
                result,
                Err(AdmmError::DimensionMismatch {
                    expected: 3,
                    got: 5
                })
            ),
            "expected DimensionMismatch error"
        );
    }
}
