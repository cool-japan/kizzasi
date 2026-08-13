//! Constraint decomposition for large-scale optimization problems
//!
//! This module provides algorithms for decomposing large constraint satisfaction
//! and optimization problems into smaller, more manageable subproblems that can be
//! solved efficiently, potentially in parallel.
//!
//! # Key Concepts
//!
//! - **Consensus ADMM**: Decomposes problems with separable objectives and coupled constraints
//! - **Dual Decomposition**: Exploits separability in the constraint structure
//! - **Block Decomposition**: Divides variables into blocks for coordinate descent
//! - **Hierarchical Decomposition**: Multi-level decomposition for very large problems

use crate::lp_simplex::{solve_ge_lp, LpOutcome};
use scirs2_core::ndarray::{Array1, Array2};

/// Result type for decomposition operations
pub type DecompositionResult<T> = Result<T, DecompositionError>;

/// Iteration budget for a simplex solve of the given size.
///
/// Generous enough that only genuinely pathological instances hit it, while
/// still guaranteeing termination.
fn simplex_budget(num_vars: usize, num_rows: usize) -> usize {
    50 * (num_vars + num_rows + 1) + 200
}

/// Convert an `f64` solver vector into the crate's `f32` representation.
fn to_f32_array(values: &[f64]) -> Array1<f32> {
    Array1::from_vec(values.iter().map(|&v| v as f32).collect())
}

/// Dot product of two `f32` arrays, tolerant of length differences.
fn dot_f32(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Compute `Mᵀv` for an `Array2` with `columns` columns.
fn transpose_dot(matrix: &Array2<f32>, vector: &Array1<f32>, columns: usize) -> Array1<f32> {
    let rows = matrix.nrows();
    let values: Vec<f32> = (0..columns)
        .map(|j| {
            (0..rows)
                .map(|i| {
                    matrix.get((i, j)).copied().unwrap_or(0.0)
                        * vector.get(i).copied().unwrap_or(0.0)
                })
                .sum()
        })
        .collect();
    Array1::from_vec(values)
}

/// Errors that can occur during constraint decomposition
#[derive(Debug, Clone)]
pub enum DecompositionError {
    /// The problem structure is not compatible with the decomposition method
    IncompatibleStructure(String),
    /// Convergence failed after maximum iterations
    ConvergenceFailed { iterations: usize, residual: f32 },
    /// Invalid block structure
    InvalidBlocks(String),
    /// Numerical issues during decomposition
    NumericalIssue(String),
}

impl std::fmt::Display for DecompositionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IncompatibleStructure(msg) => write!(f, "Incompatible structure: {}", msg),
            Self::ConvergenceFailed {
                iterations,
                residual,
            } => {
                write!(
                    f,
                    "Failed to converge after {} iterations (residual: {})",
                    iterations, residual
                )
            }
            Self::InvalidBlocks(msg) => write!(f, "Invalid blocks: {}", msg),
            Self::NumericalIssue(msg) => write!(f, "Numerical issue: {}", msg),
        }
    }
}

impl std::error::Error for DecompositionError {}

/// Block specification for block-wise decomposition
#[derive(Debug, Clone)]
pub struct Block {
    /// Name of this block
    pub name: String,
    /// Indices of variables in this block
    pub indices: Vec<usize>,
}

impl Block {
    /// Create a new block
    pub fn new(name: impl Into<String>, indices: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            indices,
        }
    }

    /// Number of variables in this block
    pub fn size(&self) -> usize {
        self.indices.len()
    }
}

/// Configuration for ADMM-based consensus optimization
#[derive(Debug, Clone)]
pub struct ADMMConfig {
    /// Penalty parameter ρ for augmented Lagrangian
    pub rho: f32,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance for primal residual
    pub primal_tol: f32,
    /// Convergence tolerance for dual residual
    pub dual_tol: f32,
    /// Whether to use adaptive ρ
    pub adaptive_rho: bool,
    /// Scaling factor for adaptive ρ updates
    pub rho_update_factor: f32,
}

impl Default for ADMMConfig {
    fn default() -> Self {
        Self {
            rho: 1.0,
            max_iterations: 1000,
            primal_tol: 1e-4,
            dual_tol: 1e-4,
            adaptive_rho: true,
            rho_update_factor: 2.0,
        }
    }
}

/// Consensus ADMM solver for decomposed optimization
///
/// Solves problems of the form:
/// ```text
/// minimize   Σᵢ fᵢ(xᵢ)
/// subject to xᵢ = z  for all i
///            z ∈ C
/// ```
///
/// where each fᵢ is a separable objective and C is a constraint set.
pub struct ConsensusADMM {
    config: ADMMConfig,
    num_blocks: usize,
    dimension: usize,
    /// Local variables xᵢ for each block
    local_vars: Vec<Array1<f32>>,
    /// Global consensus variable z
    global_var: Array1<f32>,
    /// Dual variables (scaled Lagrange multipliers)
    dual_vars: Vec<Array1<f32>>,
}

impl ConsensusADMM {
    /// Create a new consensus ADMM solver
    pub fn new(num_blocks: usize, dimension: usize, config: ADMMConfig) -> Self {
        let local_vars = vec![Array1::zeros(dimension); num_blocks];
        let global_var = Array1::zeros(dimension);
        let dual_vars = vec![Array1::zeros(dimension); num_blocks];

        Self {
            config,
            num_blocks,
            dimension,
            local_vars,
            global_var,
            dual_vars,
        }
    }

    /// Initialize from a starting point
    ///
    /// # Errors
    ///
    /// [`DecompositionError::IncompatibleStructure`] when `x0` does not have
    /// the solver's dimension.
    pub fn initialize(&mut self, x0: &Array1<f32>) -> DecompositionResult<()> {
        if x0.len() != self.dimension {
            return Err(DecompositionError::IncompatibleStructure(format!(
                "initial point must have {} entries, got {}",
                self.dimension,
                x0.len()
            )));
        }
        self.global_var = x0.clone();
        for local in self.local_vars.iter_mut() {
            *local = x0.clone();
        }
        Ok(())
    }

    /// Perform one ADMM iteration with custom local updates
    ///
    /// The `local_update_fn` should solve:
    /// ```text
    /// xᵢ := argmin fᵢ(xᵢ) + (ρ/2)‖xᵢ - z + uᵢ‖²
    /// ```
    pub fn iterate<F>(&mut self, local_update_fn: F) -> (f32, f32)
    where
        F: Fn(usize, &Array1<f32>, &Array1<f32>, f32) -> Array1<f32>,
    {
        // Step 1: Update local variables xᵢ (can be done in parallel)
        for i in 0..self.num_blocks {
            let z_minus_u = &self.global_var - &self.dual_vars[i];
            self.local_vars[i] =
                local_update_fn(i, &self.local_vars[i], &z_minus_u, self.config.rho);
        }

        // Step 2: Update global variable z (averaging + projection)
        let mut z_new = Array1::zeros(self.dimension);
        for i in 0..self.num_blocks {
            z_new += &(&self.local_vars[i] + &self.dual_vars[i]);
        }
        z_new /= self.num_blocks as f32;

        // Step 3: Update dual variables uᵢ
        let mut primal_residual = 0.0f32;

        // Per Boyd et al. (2010), the dual residual is ρ‖z^{k+1} - z^k‖ — a single
        // vector norm computed once, not accumulated per block.
        let z_diff = &z_new - &self.global_var;
        let dual_sq: f32 = z_diff.iter().map(|&x| x * x).sum();
        for i in 0..self.num_blocks {
            let xi_minus_z = &self.local_vars[i] - &z_new;
            self.dual_vars[i] = &self.dual_vars[i] + &xi_minus_z;
            primal_residual += xi_minus_z.iter().map(|&x| x * x).sum::<f32>();
        }
        let dual_residual = dual_sq;

        self.global_var = z_new;

        (
            primal_residual.sqrt(),
            dual_residual.sqrt() * self.config.rho,
        )
    }

    /// Get the current consensus solution
    pub fn solution(&self) -> &Array1<f32> {
        &self.global_var
    }

    /// Get local variable for a specific block
    pub fn local_solution(&self, block_id: usize) -> Option<&Array1<f32>> {
        self.local_vars.get(block_id)
    }

    /// Check convergence based on residuals
    pub fn has_converged(&self, primal_res: f32, dual_res: f32) -> bool {
        primal_res < self.config.primal_tol && dual_res < self.config.dual_tol
    }

    /// Update ρ adaptively based on residuals
    pub fn update_rho(&mut self, primal_res: f32, dual_res: f32) {
        if !self.config.adaptive_rho {
            return;
        }

        if primal_res > 10.0 * dual_res {
            self.config.rho *= self.config.rho_update_factor;
        } else if dual_res > 10.0 * primal_res {
            self.config.rho /= self.config.rho_update_factor;
        }
    }
}

/// Block coordinate descent for structured constraints
///
/// Solves optimization problems by iteratively optimizing over blocks of variables
/// while keeping other blocks fixed.
pub struct BlockCoordinateDescent {
    blocks: Vec<Block>,
    dimension: usize,
    current_solution: Array1<f32>,
    max_iterations: usize,
    tolerance: f32,
}

impl BlockCoordinateDescent {
    /// Create a new block coordinate descent solver
    pub fn new(blocks: Vec<Block>, dimension: usize) -> DecompositionResult<Self> {
        // Validate blocks
        let mut covered = vec![false; dimension];
        for block in &blocks {
            for &idx in &block.indices {
                if idx >= dimension {
                    return Err(DecompositionError::InvalidBlocks(format!(
                        "Index {} exceeds dimension {}",
                        idx, dimension
                    )));
                }
                if covered[idx] {
                    return Err(DecompositionError::InvalidBlocks(format!(
                        "Index {} appears in multiple blocks",
                        idx
                    )));
                }
                covered[idx] = true;
            }
        }

        Ok(Self {
            blocks,
            dimension,
            current_solution: Array1::zeros(dimension),
            max_iterations: 1000,
            tolerance: 1e-4,
        })
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Initialize from a starting point
    ///
    /// # Errors
    ///
    /// [`DecompositionError::IncompatibleStructure`] when `x0` does not have
    /// the solver's dimension.
    pub fn initialize(&mut self, x0: &Array1<f32>) -> DecompositionResult<()> {
        if x0.len() != self.dimension {
            return Err(DecompositionError::IncompatibleStructure(format!(
                "initial point must have {} entries, got {}",
                self.dimension,
                x0.len()
            )));
        }
        self.current_solution = x0.clone();
        Ok(())
    }

    /// Perform one block update
    ///
    /// The `block_update_fn` should optimize the objective over the specified block
    /// while keeping other variables fixed.
    ///
    /// # Errors
    ///
    /// [`DecompositionError::InvalidBlocks`] for an unknown `block_id`, and
    /// [`DecompositionError::IncompatibleStructure`] when the callback returns
    /// a vector of the wrong length.
    pub fn update_block<F>(
        &mut self,
        block_id: usize,
        block_update_fn: F,
    ) -> DecompositionResult<f32>
    where
        F: Fn(&Array1<f32>, &[usize]) -> Array1<f32>,
    {
        let Some(block) = self.blocks.get(block_id) else {
            return Err(DecompositionError::InvalidBlocks(format!(
                "block {block_id} does not exist (there are {} blocks)",
                self.blocks.len()
            )));
        };
        let indices = block.indices.clone();
        let block_solution = block_update_fn(&self.current_solution, &indices);

        if block_solution.len() != indices.len() {
            return Err(DecompositionError::IncompatibleStructure(format!(
                "block update must return {} values, got {}",
                indices.len(),
                block_solution.len()
            )));
        }

        // Update the solution for this block
        let mut change = 0.0f32;
        for (i, &idx) in indices.iter().enumerate() {
            let (Some(old_val), Some(&new_val)) = (
                self.current_solution.get(idx).copied(),
                block_solution.get(i),
            ) else {
                return Err(DecompositionError::InvalidBlocks(format!(
                    "block index {idx} exceeds the solution dimension {}",
                    self.current_solution.len()
                )));
            };
            if let Some(slot) = self.current_solution.get_mut(idx) {
                *slot = new_val;
            }
            change += (new_val - old_val).powi(2);
        }

        Ok(change.sqrt())
    }

    /// Get current solution
    pub fn solution(&self) -> &Array1<f32> {
        &self.current_solution
    }

    /// Get number of blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get block information
    pub fn block(&self, block_id: usize) -> Option<&Block> {
        self.blocks.get(block_id)
    }
}

/// Dual decomposition for separable constraints
///
/// Exploits problem structure where constraints can be decomposed into
/// independent subproblems coordinated through dual variables.
pub struct DualDecomposition {
    num_subproblems: usize,
    coupling_matrix: Array2<f32>,
    dual_vars: Array1<f32>,
    step_size: f32,
    max_iterations: usize,
}

impl DualDecomposition {
    /// Create a new dual decomposition solver
    ///
    /// # Arguments
    /// * `num_subproblems` - Number of decomposed subproblems
    /// * `coupling_matrix` - Matrix describing how subproblems are coupled
    pub fn new(num_subproblems: usize, coupling_matrix: Array2<f32>) -> Self {
        let num_coupling = coupling_matrix.nrows();
        Self {
            num_subproblems,
            coupling_matrix,
            dual_vars: Array1::zeros(num_coupling),
            step_size: 0.1,
            max_iterations: 1000,
        }
    }

    /// Set the dual step size
    pub fn with_step_size(mut self, step_size: f32) -> Self {
        self.step_size = step_size;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Update dual variables using subgradient method
    ///
    /// # Errors
    ///
    /// [`DecompositionError::IncompatibleStructure`] when the violation vector
    /// does not have one entry per coupling constraint.
    pub fn update_duals(&mut self, constraint_violations: &Array1<f32>) -> DecompositionResult<()> {
        if constraint_violations.len() != self.dual_vars.len() {
            return Err(DecompositionError::IncompatibleStructure(format!(
                "violation vector must have {} entries, got {}",
                self.dual_vars.len(),
                constraint_violations.len()
            )));
        }

        // Dual ascent: λ := λ + α·g(x) where g(x) is the constraint violation
        let step_size = self.step_size;
        for (dual, &violation) in self.dual_vars.iter_mut().zip(constraint_violations.iter()) {
            *dual = (*dual + step_size * violation).max(0.0);
        }
        Ok(())
    }

    /// Get current dual variables
    pub fn dual_variables(&self) -> &Array1<f32> {
        &self.dual_vars
    }

    /// Compute augmented cost for subproblem i
    ///
    /// # Errors
    ///
    /// [`DecompositionError::InvalidBlocks`] when `subproblem_id` is out of
    /// range for the configured coupling matrix.
    pub fn augmented_cost(
        &self,
        subproblem_id: usize,
        base_cost: f32,
        local_vars: &Array1<f32>,
    ) -> DecompositionResult<f32> {
        if subproblem_id >= self.num_subproblems || subproblem_id >= self.coupling_matrix.ncols() {
            return Err(DecompositionError::InvalidBlocks(format!(
                "subproblem {subproblem_id} does not exist (there are {} subproblems, \
                 coupling matrix has {} columns)",
                self.num_subproblems,
                self.coupling_matrix.ncols()
            )));
        }

        // Add dual contribution: f(x) + λᵀAx
        let mut augmented = base_cost;
        for (i, &dual) in self.dual_vars.iter().enumerate() {
            let coupling_coeff = self
                .coupling_matrix
                .get((i, subproblem_id))
                .copied()
                .unwrap_or(0.0);
            if let Some(&var_val) = local_vars.first() {
                augmented += dual * coupling_coeff * var_val;
            }
        }

        Ok(augmented)
    }
}

/// Hierarchical decomposition for very large-scale problems
///
/// Organizes constraints into a hierarchy where higher-level constraints
/// coordinate lower-level subproblems.
pub struct HierarchicalDecomposition {
    levels: Vec<DecompositionLevel>,
}

/// A single level in the hierarchical decomposition
#[derive(Clone)]
pub struct DecompositionLevel {
    /// Name of this level
    pub name: String,
    /// Subproblems at this level
    pub subproblems: Vec<Subproblem>,
}

/// A subproblem in hierarchical decomposition
#[derive(Clone)]
pub struct Subproblem {
    /// Unique identifier
    pub id: usize,
    /// Variable indices controlled by this subproblem
    pub variables: Vec<usize>,
    /// Child subproblems (at next level down)
    pub children: Vec<usize>,
}

impl HierarchicalDecomposition {
    /// Create a new hierarchical decomposition
    pub fn new() -> Self {
        Self { levels: Vec::new() }
    }

    /// Add a level to the hierarchy
    pub fn add_level(&mut self, level: DecompositionLevel) {
        self.levels.push(level);
    }

    /// Get number of levels
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get level by index
    pub fn level(&self, level_id: usize) -> Option<&DecompositionLevel> {
        self.levels.get(level_id)
    }

    /// Create a two-level decomposition from blocks
    pub fn from_blocks(
        coarse_blocks: Vec<Block>,
        fine_blocks: Vec<Block>,
    ) -> DecompositionResult<Self> {
        let mut decomp = Self::new();

        // Create fine level (bottom)
        let fine_subproblems: Vec<Subproblem> = fine_blocks
            .into_iter()
            .enumerate()
            .map(|(id, block)| Subproblem {
                id,
                variables: block.indices,
                children: Vec::new(),
            })
            .collect();

        let fine_level = DecompositionLevel {
            name: "fine".to_string(),
            subproblems: fine_subproblems,
        };

        // Create coarse level (top)
        let coarse_subproblems: Vec<Subproblem> = coarse_blocks
            .into_iter()
            .enumerate()
            .map(|(id, block)| Subproblem {
                id,
                variables: block.indices,
                children: Vec::new(), // Could link to fine subproblems
            })
            .collect();

        let coarse_level = DecompositionLevel {
            name: "coarse".to_string(),
            subproblems: coarse_subproblems,
        };

        decomp.add_level(fine_level);
        decomp.add_level(coarse_level);

        Ok(decomp)
    }
}

impl Default for HierarchicalDecomposition {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for creating block structures
pub mod block_utils {
    use super::*;

    /// Create uniform blocks of equal size
    pub fn uniform_blocks(dimension: usize, block_size: usize) -> Vec<Block> {
        let num_blocks = dimension.div_ceil(block_size);
        let mut blocks = Vec::new();

        for i in 0..num_blocks {
            let start = i * block_size;
            let end = (start + block_size).min(dimension);
            let indices: Vec<usize> = (start..end).collect();
            blocks.push(Block::new(format!("block_{}", i), indices));
        }

        blocks
    }

    /// Create blocks from explicit index groups
    pub fn from_index_groups(groups: Vec<Vec<usize>>) -> Vec<Block> {
        groups
            .into_iter()
            .enumerate()
            .map(|(i, indices)| Block::new(format!("block_{}", i), indices))
            .collect()
    }

    /// Create overlapping blocks with specified overlap
    ///
    /// # Errors
    ///
    /// [`DecompositionError::InvalidBlocks`] when `block_size` is zero or the
    /// overlap is not strictly smaller than the block size (either would make
    /// the stride zero and never terminate).
    pub fn overlapping_blocks(
        dimension: usize,
        block_size: usize,
        overlap: usize,
    ) -> DecompositionResult<Vec<Block>> {
        if block_size == 0 {
            return Err(DecompositionError::InvalidBlocks(
                "block size must be at least 1".to_string(),
            ));
        }
        if overlap >= block_size {
            return Err(DecompositionError::InvalidBlocks(format!(
                "overlap ({overlap}) must be smaller than the block size ({block_size})"
            )));
        }

        let stride = block_size - overlap;
        let mut blocks = Vec::new();
        let mut start = 0;

        while start < dimension {
            let end = (start + block_size).min(dimension);
            let indices: Vec<usize> = (start..end).collect();
            blocks.push(Block::new(format!("block_{}", blocks.len()), indices));
            start += stride;
        }

        Ok(blocks)
    }
}

// ============================================================================
// Benders Decomposition
// ============================================================================

/// Benders Decomposition for two-stage linear programs
///
/// Decomposes problems of the form:
/// ```text
/// min c'x + d'y
/// s.t. Ax + By >= b
///      lo <= x <= hi,  y >= 0
/// ```
///
/// Into:
/// - Master problem: involves only `x`, plus the Benders cuts accumulated so far
/// - Subproblem: involves only `y`, for a fixed `x`
///
/// # Algorithm
///
/// 1. Solve the relaxed master problem → get x*
/// 2. Solve the subproblem with fixed x* → get its dual variables (or, when it
///    is infeasible, a Farkas extreme ray)
/// 3. Add the corresponding optimality/feasibility cut to the master
/// 4. Repeat until the bound gap closes
///
/// Both stages are solved by the crate's own dense two-phase simplex, so the
/// duals used to build cuts are real multipliers and are checked for dual
/// feasibility before a cut is generated.
///
/// # Integrality is not enforced
///
/// The master problem is solved as a **continuous** LP over the box
/// `[lo, hi]` — by default `[0, 1]`, the binary relaxation — together with the
/// accumulated cuts. Integer restrictions are *not* imposed; enforcing them
/// requires branch-and-bound around this loop, which this crate does not
/// provide. The reported `lower_bound` is therefore the LP-relaxation bound of
/// the two-stage problem, and it is exact for problems whose master variables
/// are genuinely continuous.
pub struct BendersDecomposition {
    /// Number of master variables (first stage)
    num_master_vars: usize,
    /// Number of subproblem variables (continuous)
    num_sub_vars: usize,
    /// Benders cuts added so far
    cuts: Vec<BendersCut>,
    /// Configuration
    config: BendersConfig,
    /// Current lower bound
    lower_bound: f32,
    /// Current upper bound
    upper_bound: f32,
    /// Lower box bound of the master variables
    master_lower: Array1<f32>,
    /// Upper box bound of the master variables
    master_upper: Array1<f32>,
}

/// Solution of a Benders subproblem
#[derive(Debug, Clone)]
pub struct BendersSubproblemSolution {
    /// Optimal subproblem objective `d'y` (`+∞` when infeasible)
    pub objective: f32,
    /// Optimal subproblem variables `y` (all zeros when infeasible)
    pub primal: Array1<f32>,
    /// Dual multipliers of `By >= b - Ax`, or the Farkas extreme ray when the
    /// subproblem is infeasible
    pub dual: Array1<f32>,
    /// Whether the subproblem admitted a feasible `y`
    pub is_feasible: bool,
}

/// Configuration for Benders decomposition
#[derive(Debug, Clone)]
pub struct BendersConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance (gap between bounds)
    pub tolerance: f32,
    /// Maximum number of cuts to keep
    pub max_cuts: usize,
    /// Whether to use multi-cut strategy
    pub multi_cut: bool,
}

impl Default for BendersConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-4,
            max_cuts: 1000,
            multi_cut: false,
        }
    }
}

/// Benders cut: either optimality or feasibility
#[derive(Debug, Clone)]
pub enum BendersCut {
    /// Optimality cut: θ >= f(x) + π'(b - Ax)
    /// where π are dual variables from subproblem
    Optimality {
        /// Dual variables (multipliers)
        dual: Array1<f32>,
        /// Constant term
        constant: f32,
        /// Coefficient matrix for master variables
        coefficients: Array1<f32>,
    },
    /// Feasibility cut: ensures subproblem is feasible
    /// π'(b - Ax) >= 0 where π are extreme rays
    Feasibility {
        /// Extreme ray
        ray: Array1<f32>,
        /// Constant term
        constant: f32,
        /// Coefficient matrix
        coefficients: Array1<f32>,
    },
}

impl BendersDecomposition {
    /// Create a new Benders decomposition
    ///
    /// Master variables default to the box `[0, 1]` (the binary relaxation);
    /// use [`Self::with_master_bounds`] for a different first-stage domain.
    pub fn new(num_master_vars: usize, num_sub_vars: usize) -> Self {
        Self {
            num_master_vars,
            num_sub_vars,
            cuts: Vec::new(),
            config: BendersConfig::default(),
            lower_bound: f32::NEG_INFINITY,
            upper_bound: f32::INFINITY,
            master_lower: Array1::zeros(num_master_vars),
            master_upper: Array1::from_elem(num_master_vars, 1.0),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: BendersConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the box bounds of the master variables
    ///
    /// # Errors
    ///
    /// Returns [`DecompositionError::IncompatibleStructure`] when the vectors
    /// do not have `num_master_vars` entries or when some `lower[i] > upper[i]`.
    pub fn with_master_bounds(
        mut self,
        lower: Array1<f32>,
        upper: Array1<f32>,
    ) -> DecompositionResult<Self> {
        if lower.len() != self.num_master_vars || upper.len() != self.num_master_vars {
            return Err(DecompositionError::IncompatibleStructure(format!(
                "master bounds must have {} entries, got {} and {}",
                self.num_master_vars,
                lower.len(),
                upper.len()
            )));
        }
        for (index, (lo, hi)) in lower.iter().zip(upper.iter()).enumerate() {
            if lo > hi {
                return Err(DecompositionError::IncompatibleStructure(format!(
                    "master bound {index} is empty: lower {lo} exceeds upper {hi}"
                )));
            }
        }
        self.master_lower = lower;
        self.master_upper = upper;
        Ok(self)
    }

    /// Current master variable bounds `(lower, upper)`
    pub fn master_bounds(&self) -> (&Array1<f32>, &Array1<f32>) {
        (&self.master_lower, &self.master_upper)
    }

    /// Add an optimality cut
    pub fn add_optimality_cut(
        &mut self,
        dual: Array1<f32>,
        constant: f32,
        coefficients: Array1<f32>,
    ) {
        if self.cuts.len() >= self.config.max_cuts {
            // Remove oldest cut if at limit
            self.cuts.remove(0);
        }

        self.cuts.push(BendersCut::Optimality {
            dual,
            constant,
            coefficients,
        });
    }

    /// Add a feasibility cut
    pub fn add_feasibility_cut(
        &mut self,
        ray: Array1<f32>,
        constant: f32,
        coefficients: Array1<f32>,
    ) {
        if self.cuts.len() >= self.config.max_cuts {
            self.cuts.remove(0);
        }

        self.cuts.push(BendersCut::Feasibility {
            ray,
            constant,
            coefficients,
        });
    }

    /// Solve the master problem
    ///
    /// Minimises `c'x + θ` over the master box bounds subject to every
    /// accumulated cut:
    ///
    /// ```text
    /// θ >= constant_k + coefficients_k · x     (optimality cuts)
    /// 0 <= constant_j + coefficients_j · x     (feasibility cuts)
    /// lo <= x <= hi
    /// ```
    ///
    /// The LP is solved by the crate's dense two-phase simplex. When no
    /// optimality cut has been generated yet, `θ` is unconstrained from below,
    /// so the returned lower bound is `-∞` (the master carries no information
    /// about the recourse cost yet) and only `c'x` is minimised.
    ///
    /// Returns the master solution and the corresponding lower bound.
    ///
    /// # Errors
    ///
    /// * [`DecompositionError::IncompatibleStructure`] when there are no master
    ///   variables, `objective` has the wrong length, or the accumulated
    ///   feasibility cuts leave no feasible `x`.
    /// * [`DecompositionError::NumericalIssue`] when the LP is unbounded or the
    ///   simplex cannot produce consistent multipliers.
    pub fn solve_master(&self, objective: &Array1<f32>) -> DecompositionResult<(Array1<f32>, f32)> {
        if self.num_master_vars == 0 {
            return Err(DecompositionError::IncompatibleStructure(
                "No master variables".to_string(),
            ));
        }
        if objective.len() != self.num_master_vars {
            return Err(DecompositionError::IncompatibleStructure(format!(
                "master objective must have {} entries, got {}",
                self.num_master_vars,
                objective.len()
            )));
        }

        let n = self.num_master_vars;
        let lower: Vec<f64> = self.master_lower.iter().map(|&v| v as f64).collect();
        let width: Vec<f64> = self
            .master_lower
            .iter()
            .zip(self.master_upper.iter())
            .map(|(&lo, &hi)| (hi - lo) as f64)
            .collect();

        let has_theta = self
            .cuts
            .iter()
            .any(|cut| matches!(cut, BendersCut::Optimality { .. }));

        // Shifted variables x' = x - lo >= 0, plus θ = θ⁺ - θ⁻ when needed.
        let num_vars = if has_theta { n + 2 } else { n };
        let mut costs = vec![0.0_f64; num_vars];
        for (j, &value) in objective.iter().enumerate() {
            if let Some(slot) = costs.get_mut(j) {
                *slot = value as f64;
            }
        }
        if has_theta {
            if let Some(slot) = costs.get_mut(n) {
                *slot = 1.0;
            }
            if let Some(slot) = costs.get_mut(n + 1) {
                *slot = -1.0;
            }
        }

        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut rhs: Vec<f64> = Vec::new();

        // Upper bounds: -x'_j >= -(hi_j - lo_j)
        for (j, &span) in width.iter().enumerate() {
            let mut row = vec![0.0_f64; num_vars];
            if let Some(slot) = row.get_mut(j) {
                *slot = -1.0;
            }
            rows.push(row);
            rhs.push(-span);
        }

        for cut in &self.cuts {
            match cut {
                BendersCut::Optimality {
                    constant,
                    coefficients,
                    ..
                } => {
                    // θ - coefficients·x' >= constant + coefficients·lo
                    let mut row = vec![0.0_f64; num_vars];
                    let mut shift = 0.0_f64;
                    for j in 0..n {
                        let coefficient = coefficients.get(j).copied().unwrap_or(0.0) as f64;
                        if let Some(slot) = row.get_mut(j) {
                            *slot = -coefficient;
                        }
                        shift += coefficient * lower.get(j).copied().unwrap_or(0.0);
                    }
                    if let Some(slot) = row.get_mut(n) {
                        *slot = 1.0;
                    }
                    if let Some(slot) = row.get_mut(n + 1) {
                        *slot = -1.0;
                    }
                    rows.push(row);
                    rhs.push(*constant as f64 + shift);
                }
                BendersCut::Feasibility {
                    constant,
                    coefficients,
                    ..
                } => {
                    // coefficients·x' >= -constant - coefficients·lo
                    let mut row = vec![0.0_f64; num_vars];
                    let mut shift = 0.0_f64;
                    for j in 0..n {
                        let coefficient = coefficients.get(j).copied().unwrap_or(0.0) as f64;
                        if let Some(slot) = row.get_mut(j) {
                            *slot = coefficient;
                        }
                        shift += coefficient * lower.get(j).copied().unwrap_or(0.0);
                    }
                    rows.push(row);
                    rhs.push(-(*constant as f64) - shift);
                }
            }
        }

        let simplex_budget = simplex_budget(num_vars, rows.len());
        match solve_ge_lp(&costs, &rows, &rhs, simplex_budget) {
            LpOutcome::Optimal {
                primal,
                objective: lp_objective,
                ..
            } => {
                let x: Array1<f32> = (0..n)
                    .map(|j| {
                        (lower.get(j).copied().unwrap_or(0.0)
                            + primal.get(j).copied().unwrap_or(0.0)) as f32
                    })
                    .collect::<Vec<f32>>()
                    .into();

                let lower_bound = if has_theta {
                    let shift: f64 = (0..n)
                        .map(|j| {
                            objective.get(j).copied().unwrap_or(0.0) as f64
                                * lower.get(j).copied().unwrap_or(0.0)
                        })
                        .sum();
                    (lp_objective + shift) as f32
                } else {
                    // No optimality cut yet: θ is unbounded below, so the master
                    // provides no finite lower bound.
                    f32::NEG_INFINITY
                };

                Ok((x, lower_bound))
            }
            LpOutcome::Infeasible { .. } => Err(DecompositionError::IncompatibleStructure(
                "master problem is infeasible: the accumulated feasibility cuts and the master \
                 bounds have no common point"
                    .to_string(),
            )),
            LpOutcome::Unbounded => Err(DecompositionError::NumericalIssue(
                "master problem is unbounded below; tighten the master bounds".to_string(),
            )),
            LpOutcome::IterationLimit => Err(DecompositionError::NumericalIssue(
                "master LP exhausted the simplex iteration budget".to_string(),
            )),
            LpOutcome::NumericalFailure(reason) => Err(DecompositionError::NumericalIssue(
                format!("master LP failed: {reason}"),
            )),
        }
    }

    /// Solve the subproblem for a fixed master solution
    ///
    /// Solves `min d'y  s.t.  By >= b - Ax*,  y >= 0` with the crate's dense
    /// two-phase simplex, where `A` is `master_coupling` and `B` is
    /// `sub_coupling`.
    ///
    /// When the subproblem is feasible the returned `dual` holds genuine
    /// multipliers of `By >= b - Ax*` (checked against `B'π <= d`, `π >= 0`).
    /// When it is infeasible, `is_feasible` is `false` and `dual` holds a
    /// Farkas extreme ray (`π >= 0`, `B'π <= 0`, `π'(b - Ax*) > 0`) suitable
    /// for a feasibility cut.
    ///
    /// # Errors
    ///
    /// * [`DecompositionError::IncompatibleStructure`] on any shape mismatch.
    /// * [`DecompositionError::NumericalIssue`] when the subproblem objective
    ///   is unbounded below, or the simplex cannot produce consistent
    ///   multipliers.
    pub fn solve_subproblem(
        &self,
        master_solution: &Array1<f32>,
        sub_objective: &Array1<f32>,
        master_coupling: &Array2<f32>,
        sub_coupling: &Array2<f32>,
        rhs: &Array1<f32>,
    ) -> DecompositionResult<BendersSubproblemSolution> {
        let (n_constraints, n_sub_vars) = sub_coupling.dim();
        let (n_master_rows, n_master_cols) = master_coupling.dim();

        if n_sub_vars != self.num_sub_vars {
            return Err(DecompositionError::IncompatibleStructure(format!(
                "sub coupling matrix must have {} columns, got {}",
                self.num_sub_vars, n_sub_vars
            )));
        }
        if n_master_cols != self.num_master_vars {
            return Err(DecompositionError::IncompatibleStructure(format!(
                "master coupling matrix must have {} columns, got {}",
                self.num_master_vars, n_master_cols
            )));
        }
        if n_master_rows != n_constraints {
            return Err(DecompositionError::IncompatibleStructure(format!(
                "coupling matrices disagree on the constraint count: {n_master_rows} vs {n_constraints}"
            )));
        }
        if rhs.len() != n_constraints {
            return Err(DecompositionError::IncompatibleStructure(format!(
                "right-hand side must have {n_constraints} entries, got {}",
                rhs.len()
            )));
        }
        if master_solution.len() != self.num_master_vars {
            return Err(DecompositionError::IncompatibleStructure(format!(
                "master solution must have {} entries, got {}",
                self.num_master_vars,
                master_solution.len()
            )));
        }
        if sub_objective.len() != self.num_sub_vars {
            return Err(DecompositionError::IncompatibleStructure(format!(
                "sub objective must have {} entries, got {}",
                self.num_sub_vars,
                sub_objective.len()
            )));
        }

        // Adjusted right-hand side: b - Ax*
        let adjusted: Vec<f64> = (0..n_constraints)
            .map(|i| {
                let coupled: f64 = (0..self.num_master_vars)
                    .map(|j| {
                        master_coupling.get((i, j)).copied().unwrap_or(0.0) as f64
                            * master_solution.get(j).copied().unwrap_or(0.0) as f64
                    })
                    .sum();
                rhs.get(i).copied().unwrap_or(0.0) as f64 - coupled
            })
            .collect();

        let matrix: Vec<Vec<f64>> = (0..n_constraints)
            .map(|i| {
                (0..n_sub_vars)
                    .map(|j| sub_coupling.get((i, j)).copied().unwrap_or(0.0) as f64)
                    .collect()
            })
            .collect();
        let costs: Vec<f64> = sub_objective.iter().map(|&v| v as f64).collect();

        let budget = simplex_budget(n_sub_vars, n_constraints);
        match solve_ge_lp(&costs, &matrix, &adjusted, budget) {
            LpOutcome::Optimal {
                primal,
                dual,
                objective,
            } => Ok(BendersSubproblemSolution {
                objective: objective as f32,
                primal: to_f32_array(&primal),
                dual: to_f32_array(&dual),
                is_feasible: true,
            }),
            LpOutcome::Infeasible { farkas } => Ok(BendersSubproblemSolution {
                objective: f32::INFINITY,
                primal: Array1::zeros(self.num_sub_vars),
                dual: to_f32_array(&farkas),
                is_feasible: false,
            }),
            LpOutcome::Unbounded => Err(DecompositionError::NumericalIssue(
                "subproblem objective is unbounded below for this master solution".to_string(),
            )),
            LpOutcome::IterationLimit => Err(DecompositionError::NumericalIssue(
                "subproblem LP exhausted the simplex iteration budget".to_string(),
            )),
            LpOutcome::NumericalFailure(reason) => Err(DecompositionError::NumericalIssue(
                format!("subproblem LP failed: {reason}"),
            )),
        }
    }

    /// Run the Benders loop until the bound gap closes
    ///
    /// `master_coupling` is `A` and `sub_coupling` is `B` in
    /// `Ax + By >= rhs`. Cuts are derived from the subproblem's real
    /// multipliers:
    ///
    /// ```text
    /// optimality:  θ >= π'b - (A'π)·x
    /// feasibility: (A'r)·x - r'b >= 0
    /// ```
    ///
    /// # Errors
    ///
    /// * [`DecompositionError::ConvergenceFailed`] when `config.max_iterations`
    ///   is exhausted before the gap closes.
    /// * Any error surfaced by [`Self::solve_master`] or
    ///   [`Self::solve_subproblem`].
    pub fn iterate(
        &mut self,
        master_obj: &Array1<f32>,
        sub_obj: &Array1<f32>,
        master_coupling: &Array2<f32>,
        sub_coupling: &Array2<f32>,
        rhs: &Array1<f32>,
    ) -> DecompositionResult<BendersIterationResult> {
        let mut iteration = 0;
        let mut incumbent: Option<(Array1<f32>, Array1<f32>)> = None;
        let mut incumbent_value = f32::INFINITY;

        loop {
            iteration += 1;

            if iteration > self.config.max_iterations {
                return Err(DecompositionError::ConvergenceFailed {
                    iterations: iteration - 1,
                    residual: self.upper_bound - self.lower_bound,
                });
            }

            // Step 1: Solve the master problem
            let (master_sol, lower_bound) = self.solve_master(master_obj)?;
            self.lower_bound = lower_bound;

            // Step 2: Solve the subproblem for this master solution
            let sub =
                self.solve_subproblem(&master_sol, sub_obj, master_coupling, sub_coupling, rhs)?;

            if sub.is_feasible {
                // Step 3: Update the incumbent and the upper bound
                let total = master_obj.dot(&master_sol) + sub.objective;
                if total < incumbent_value {
                    incumbent_value = total;
                    incumbent = Some((master_sol.clone(), sub.primal.clone()));
                }
                if total < self.upper_bound {
                    self.upper_bound = total;
                }

                // Step 4: Check convergence
                let gap = self.upper_bound - self.lower_bound;
                if self.lower_bound.is_finite()
                    && self.upper_bound.is_finite()
                    && gap < self.config.tolerance
                {
                    let (best_master, best_sub) = incumbent.unwrap_or((master_sol, sub.primal));
                    return Ok(BendersIterationResult {
                        master_solution: best_master,
                        sub_solution: best_sub,
                        lower_bound: self.lower_bound,
                        upper_bound: self.upper_bound,
                        iterations: iteration,
                        converged: true,
                    });
                }

                // Step 5: Optimality cut  θ >= π'b - (A'π)·x
                let constant = dot_f32(&sub.dual, rhs);
                let coefficients = transpose_dot(master_coupling, &sub.dual, self.num_master_vars);
                self.add_optimality_cut(sub.dual, constant, -coefficients);
            } else {
                // Step 5': Feasibility cut  (A'r)·x - r'b >= 0
                let constant = -dot_f32(&sub.dual, rhs);
                let coefficients = transpose_dot(master_coupling, &sub.dual, self.num_master_vars);
                self.add_feasibility_cut(sub.dual, constant, coefficients);
            }
        }
    }

    /// Get current bounds
    pub fn bounds(&self) -> (f32, f32) {
        (self.lower_bound, self.upper_bound)
    }

    /// Number of cuts generated
    pub fn num_cuts(&self) -> usize {
        self.cuts.len()
    }

    /// Get all cuts
    pub fn cuts(&self) -> &[BendersCut] {
        &self.cuts
    }

    /// Reset the decomposition
    pub fn reset(&mut self) {
        self.cuts.clear();
        self.lower_bound = f32::NEG_INFINITY;
        self.upper_bound = f32::INFINITY;
    }
}

/// Result of Benders iteration
#[derive(Debug, Clone)]
pub struct BendersIterationResult {
    /// Master problem solution
    pub master_solution: Array1<f32>,
    /// Subproblem solution
    pub sub_solution: Array1<f32>,
    /// Lower bound on optimal value
    pub lower_bound: f32,
    /// Upper bound on optimal value
    pub upper_bound: f32,
    /// Number of iterations
    pub iterations: usize,
    /// Whether algorithm converged
    pub converged: bool,
}

impl BendersIterationResult {
    /// Get optimality gap
    pub fn gap(&self) -> f32 {
        self.upper_bound - self.lower_bound
    }

    /// Get relative gap
    pub fn relative_gap(&self) -> f32 {
        if self.upper_bound.abs() < 1e-10 {
            self.gap()
        } else {
            self.gap() / self.upper_bound.abs()
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benders_decomposition() {
        let mut benders = BendersDecomposition::new(3, 2);
        assert_eq!(benders.num_cuts(), 0);

        // Add an optimality cut
        let dual = Array1::from_vec(vec![1.0, 2.0]);
        let coeffs = Array1::from_vec(vec![0.5, 0.3, 0.1]);
        benders.add_optimality_cut(dual, 1.5, coeffs);

        assert_eq!(benders.num_cuts(), 1);

        let (lb, ub) = benders.bounds();
        assert_eq!(lb, f32::NEG_INFINITY);
        assert_eq!(ub, f32::INFINITY);
    }

    #[test]
    fn test_benders_config() {
        let config = BendersConfig {
            max_iterations: 50,
            tolerance: 1e-5,
            max_cuts: 500,
            multi_cut: true,
        };

        let benders = BendersDecomposition::new(2, 3).with_config(config);
        assert_eq!(benders.config.max_iterations, 50);
    }

    #[test]
    fn test_consensus_admm_initialization() {
        let config = ADMMConfig::default();
        let mut admm = ConsensusADMM::new(3, 5, config);

        let x0 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        admm.initialize(&x0).expect("matching dimension");

        assert_eq!(admm.solution(), &x0);
    }

    #[test]
    fn test_block_coordinate_descent() {
        let blocks = vec![
            Block::new("b1", vec![0, 1]),
            Block::new("b2", vec![2, 3]),
            Block::new("b3", vec![4]),
        ];

        let bcd = BlockCoordinateDescent::new(blocks, 5).unwrap();
        assert_eq!(bcd.num_blocks(), 3);
        assert_eq!(bcd.block(0).unwrap().size(), 2);
    }

    #[test]
    fn test_invalid_blocks() {
        // Block with index out of bounds
        let blocks = vec![Block::new("b1", vec![0, 1, 10])];
        let result = BlockCoordinateDescent::new(blocks, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_uniform_blocks() {
        let blocks = block_utils::uniform_blocks(10, 3);
        assert_eq!(blocks.len(), 4); // ceil(10/3) = 4 blocks
        assert_eq!(blocks[0].size(), 3);
        assert_eq!(blocks[3].size(), 1); // Last block has only 1 element
    }

    #[test]
    fn test_overlapping_blocks() {
        let blocks = block_utils::overlapping_blocks(10, 4, 1).expect("valid overlap");
        assert!(blocks.len() > 3);

        // Check first two blocks overlap
        let b0_last = blocks[0].indices.last().unwrap();
        let b1_first = blocks[1].indices.first().unwrap();
        assert_eq!(b0_last, b1_first);
    }

    #[test]
    fn test_dual_decomposition() {
        let coupling = Array2::from_shape_vec((2, 3), vec![1.0, 0.5, 0.0, 0.0, 0.5, 1.0]).unwrap();
        let dual_decomp = DualDecomposition::new(3, coupling);

        assert_eq!(dual_decomp.dual_variables().len(), 2);
    }

    #[test]
    fn test_hierarchical_decomposition() {
        let mut hier = HierarchicalDecomposition::new();
        assert_eq!(hier.num_levels(), 0);

        let level = DecompositionLevel {
            name: "test".to_string(),
            subproblems: vec![],
        };
        hier.add_level(level);
        assert_eq!(hier.num_levels(), 1);
    }

    #[test]
    fn test_benders_feasibility_cut_satisfied_no_change() {
        // Cut: 1.0 + [1,0,0]·x >= 0.  At x0=zeros(3) this equals 1.0 >= 0: already satisfied.
        // solve_master should not alter x away from zero; the cut must remain satisfied.
        let mut benders = BendersDecomposition::new(3, 2);
        let ray = Array1::from_vec(vec![0.0, 0.0]);
        let constant = 1.0f32;
        let coefficients = Array1::from_vec(vec![1.0f32, 0.0, 0.0]);
        benders.add_feasibility_cut(ray, constant, coefficients.clone());

        let objective = Array1::from_vec(vec![1.0f32, 1.0, 1.0]);
        let (solution, _lower_bound) = benders.solve_master(&objective).unwrap();

        let cut_value = constant + coefficients.dot(&solution);
        assert!(
            cut_value >= -1e-5,
            "Feasibility cut must be satisfied; cut_value = {}",
            cut_value
        );
    }

    #[test]
    fn test_benders_feasibility_cut_violated_fixed() {
        // Cut: -1.0 + [1,0,0]·x >= 0.  At x0=zeros(3) this equals -1.0 < 0: violated.
        // After the fix, solve_master projects x along [1,0,0] to exactly restore feasibility.
        let mut benders = BendersDecomposition::new(3, 2);
        let ray = Array1::from_vec(vec![0.0, 0.0]);
        let constant = -1.0f32;
        let coefficients = Array1::from_vec(vec![1.0f32, 0.0, 0.0]);
        benders.add_feasibility_cut(ray, constant, coefficients.clone());

        let objective = Array1::from_vec(vec![1.0f32, 1.0, 1.0]);
        let (solution, _lower_bound) = benders.solve_master(&objective).unwrap();

        let cut_value = constant + coefficients.dot(&solution);
        assert!(
            cut_value >= -1e-5,
            "Violated feasibility cut must be fixed by projected gradient step; cut_value = {}",
            cut_value
        );
    }

    #[test]
    fn test_benders_multiple_cuts() {
        // Two orthogonal feasibility cuts:
        //   Cut A: -0.5 + [1,0,0]·x >= 0   (requires x[0] >= 0.5)
        //   Cut B: -0.3 + [0,1,0]·x >= 0   (requires x[1] >= 0.3)
        // Both are violated at x0=zeros(3). solve_master processes cuts sequentially,
        // so the final x must satisfy both cuts (they are independent/orthogonal).
        let mut benders = BendersDecomposition::new(3, 2);

        let constant_a = -0.5f32;
        let coefficients_a = Array1::from_vec(vec![1.0f32, 0.0, 0.0]);
        benders.add_feasibility_cut(
            Array1::from_vec(vec![0.0, 0.0]),
            constant_a,
            coefficients_a.clone(),
        );

        let constant_b = -0.3f32;
        let coefficients_b = Array1::from_vec(vec![0.0f32, 1.0, 0.0]);
        benders.add_feasibility_cut(
            Array1::from_vec(vec![0.0, 0.0]),
            constant_b,
            coefficients_b.clone(),
        );

        let objective = Array1::from_vec(vec![1.0f32, 1.0, 1.0]);
        let (solution, _lower_bound) = benders.solve_master(&objective).unwrap();

        // No panic must occur; both cuts should be approximately satisfied
        let cut_value_a = constant_a + coefficients_a.dot(&solution);
        let cut_value_b = constant_b + coefficients_b.dot(&solution);
        assert!(
            cut_value_a >= -1e-5,
            "Cut A must be satisfied; cut_value_a = {}",
            cut_value_a
        );
        assert!(
            cut_value_b >= -1e-5,
            "Cut B must be satisfied; cut_value_b = {}",
            cut_value_b
        );
    }

    /// Regression (finding 125/298): `iterate` must reach the *known* optimum
    /// of a small two-stage LP instead of reporting fabricated convergence.
    ///
    /// Problem: min x + 2y  s.t.  x + y >= 1,  x ∈ [0, 1],  y >= 0.
    /// Optimum: x = 1, y = 0, objective 1.
    #[test]
    fn test_benders_reaches_known_optimum() {
        let mut benders = BendersDecomposition::new(1, 1);

        let master_obj = Array1::from_vec(vec![1.0_f32]);
        let sub_obj = Array1::from_vec(vec![2.0_f32]);
        let master_coupling = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("A");
        let sub_coupling = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("B");
        let rhs = Array1::from_vec(vec![1.0_f32]);

        let result = benders
            .iterate(&master_obj, &sub_obj, &master_coupling, &sub_coupling, &rhs)
            .expect("benders must converge");

        assert!(result.converged);
        assert!(
            (result.master_solution[0] - 1.0).abs() < 1e-3,
            "x* should be 1.0, got {}",
            result.master_solution[0]
        );
        assert!(
            result.sub_solution[0].abs() < 1e-3,
            "y* should be 0.0, got {}",
            result.sub_solution[0]
        );
        assert!(
            (result.upper_bound - 1.0).abs() < 1e-3,
            "optimal value should be 1.0, got {}",
            result.upper_bound
        );
        assert!(
            result.gap().abs() < 1e-3,
            "converged gap must be ~0, got {}",
            result.gap()
        );
    }

    /// Regression (finding 125/298): the optimum must depend on the data.
    ///
    /// Same structure, but recourse is now cheaper than the first stage:
    /// min 3x + y  s.t. x + y >= 1 → x = 0, y = 1, objective 1.
    #[test]
    fn test_benders_optimum_follows_the_cost_data() {
        let mut benders = BendersDecomposition::new(1, 1);

        let master_obj = Array1::from_vec(vec![3.0_f32]);
        let sub_obj = Array1::from_vec(vec![1.0_f32]);
        let master_coupling = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("A");
        let sub_coupling = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("B");
        let rhs = Array1::from_vec(vec![1.0_f32]);

        let result = benders
            .iterate(&master_obj, &sub_obj, &master_coupling, &sub_coupling, &rhs)
            .expect("benders must converge");

        assert!(result.converged);
        assert!(
            result.master_solution[0].abs() < 1e-3,
            "x* should stay at 0.0, got {}",
            result.master_solution[0]
        );
        assert!(
            (result.sub_solution[0] - 1.0).abs() < 1e-3,
            "y* should be 1.0, got {}",
            result.sub_solution[0]
        );
        assert!((result.upper_bound - 1.0).abs() < 1e-3);
    }

    /// Regression (finding 125/298): an infeasible subproblem must be detected
    /// (the old code hardcoded `is_feasible = true`, so feasibility cuts were
    /// dead code).
    #[test]
    fn test_benders_detects_infeasible_subproblem() {
        let benders = BendersDecomposition::new(1, 1);

        // Subproblem: -y >= 1 - x with y >= 0. At x = 0 this needs y <= -1.
        let sub_obj = Array1::from_vec(vec![1.0_f32]);
        let master_coupling = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("A");
        let sub_coupling = Array2::from_shape_vec((1, 1), vec![-1.0_f32]).expect("B");
        let rhs = Array1::from_vec(vec![1.0_f32]);

        let infeasible = benders
            .solve_subproblem(
                &Array1::from_vec(vec![0.0_f32]),
                &sub_obj,
                &master_coupling,
                &sub_coupling,
                &rhs,
            )
            .expect("solve must not error");
        assert!(
            !infeasible.is_feasible,
            "subproblem at x=0 must be reported infeasible"
        );
        assert!(
            infeasible.dual[0] > 0.0,
            "an extreme ray must be returned, got {:?}",
            infeasible.dual
        );

        // At x = 1 the adjusted right-hand side is 0 and y = 0 is feasible.
        let feasible = benders
            .solve_subproblem(
                &Array1::from_vec(vec![1.0_f32]),
                &sub_obj,
                &master_coupling,
                &sub_coupling,
                &rhs,
            )
            .expect("solve must not error");
        assert!(feasible.is_feasible, "subproblem at x=1 must be feasible");
    }

    /// Regression (finding 125/298): the full loop must generate a feasibility
    /// cut and still converge for a problem whose first iterate is infeasible.
    #[test]
    fn test_benders_generates_feasibility_cut() {
        let mut benders = BendersDecomposition::new(1, 1);

        let master_obj = Array1::from_vec(vec![1.0_f32]);
        let sub_obj = Array1::from_vec(vec![1.0_f32]);
        let master_coupling = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("A");
        let sub_coupling = Array2::from_shape_vec((1, 1), vec![-1.0_f32]).expect("B");
        let rhs = Array1::from_vec(vec![1.0_f32]);

        let result = benders
            .iterate(&master_obj, &sub_obj, &master_coupling, &sub_coupling, &rhs)
            .expect("benders must converge");

        assert!(result.converged);
        assert!(
            (result.master_solution[0] - 1.0).abs() < 1e-3,
            "the feasibility cut forces x >= 1, got {}",
            result.master_solution[0]
        );
        assert!(
            benders
                .cuts()
                .iter()
                .any(|cut| matches!(cut, BendersCut::Feasibility { .. })),
            "an infeasible subproblem must produce a feasibility cut"
        );
    }

    /// Regression (finding 125/298): the subproblem must actually read its
    /// inputs — the old version returned `y = 0.5`, `dual = 1.0` for every
    /// input.
    #[test]
    fn test_benders_subproblem_depends_on_master_solution() {
        let benders = BendersDecomposition::new(1, 1);

        let sub_obj = Array1::from_vec(vec![2.0_f32]);
        let master_coupling = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("A");
        let sub_coupling = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("B");
        let rhs = Array1::from_vec(vec![4.0_f32]);

        let at_zero = benders
            .solve_subproblem(
                &Array1::from_vec(vec![0.0_f32]),
                &sub_obj,
                &master_coupling,
                &sub_coupling,
                &rhs,
            )
            .expect("solve at x=0");
        let at_one = benders
            .solve_subproblem(
                &Array1::from_vec(vec![1.0_f32]),
                &sub_obj,
                &master_coupling,
                &sub_coupling,
                &rhs,
            )
            .expect("solve at x=1");

        // y >= 4 - x, so y* = 4 and 3, objectives 8 and 6.
        assert!((at_zero.primal[0] - 4.0).abs() < 1e-4, "{:?}", at_zero);
        assert!((at_zero.objective - 8.0).abs() < 1e-4, "{:?}", at_zero);
        assert!((at_one.primal[0] - 3.0).abs() < 1e-4, "{:?}", at_one);
        assert!((at_one.objective - 6.0).abs() < 1e-4, "{:?}", at_one);
        // Dual of a binding y >= r row with cost 2 is exactly 2.
        assert!((at_zero.dual[0] - 2.0).abs() < 1e-4, "{:?}", at_zero);
    }

    /// Regression (finding 125): a mismatched master objective must return an
    /// error instead of panicking inside ndarray.
    #[test]
    fn test_benders_master_objective_dimension_checked() {
        let benders = BendersDecomposition::new(3, 2);
        let result = benders.solve_master(&Array1::from_vec(vec![1.0_f32, 2.0]));
        assert!(
            matches!(result, Err(DecompositionError::IncompatibleStructure(_))),
            "wrong-length objective must be rejected"
        );
    }

    /// A multi-variable instance with a non-trivial first stage.
    ///
    /// min 1·x0 + 4·x1 + 3·y  s.t. x0 + x1 + y >= 1, x ∈ [0, 1]², y >= 0.
    /// Optimum: x0 = 1 (cheapest), objective 1.
    #[test]
    fn test_benders_multi_variable_optimum() {
        let mut benders = BendersDecomposition::new(2, 1);

        let master_obj = Array1::from_vec(vec![1.0_f32, 4.0]);
        let sub_obj = Array1::from_vec(vec![3.0_f32]);
        let master_coupling = Array2::from_shape_vec((1, 2), vec![1.0_f32, 1.0]).expect("A");
        let sub_coupling = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("B");
        let rhs = Array1::from_vec(vec![1.0_f32]);

        let result = benders
            .iterate(&master_obj, &sub_obj, &master_coupling, &sub_coupling, &rhs)
            .expect("benders must converge");

        assert!(result.converged);
        assert!(
            (result.upper_bound - 1.0).abs() < 1e-3,
            "optimum should be 1.0, got {}",
            result.upper_bound
        );
        assert!(
            (result.master_solution[0] - 1.0).abs() < 1e-3,
            "the cheap first-stage variable should be used: {:?}",
            result.master_solution
        );
    }

    /// Regression (finding 140): invalid input to these public entry points
    /// must return a typed error instead of aborting through `assert!`.
    #[test]
    fn test_decomposition_entry_points_return_errors() {
        let mut admm = ConsensusADMM::new(2, 3, ADMMConfig::default());
        assert!(admm
            .initialize(&Array1::from_vec(vec![1.0_f32, 2.0]))
            .is_err());

        let mut bcd = BlockCoordinateDescent::new(vec![Block::new("b", vec![0, 1])], 2)
            .expect("valid blocks");
        assert!(bcd.initialize(&Array1::from_vec(vec![1.0_f32])).is_err());
        assert!(bcd
            .update_block(7, |_x, indices| Array1::zeros(indices.len()))
            .is_err());
        assert!(bcd
            .update_block(0, |_x, _indices| Array1::zeros(5))
            .is_err());

        let coupling =
            Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0, 0.0, 1.0]).expect("coupling matrix");
        let mut dual = DualDecomposition::new(2, coupling);
        assert!(dual.update_duals(&Array1::from_vec(vec![1.0_f32])).is_err());
        assert!(dual
            .augmented_cost(9, 0.0, &Array1::from_vec(vec![1.0_f32]))
            .is_err());

        assert!(block_utils::overlapping_blocks(10, 4, 4).is_err());
        assert!(block_utils::overlapping_blocks(10, 0, 0).is_err());
        assert!(block_utils::overlapping_blocks(10, 4, 1).is_ok());
    }

    #[test]
    fn test_admm_dual_residual_single_block_baseline() {
        // With num_blocks=1 the old and new code agree (loop runs once → same result).
        // Verifies the fix doesn't break the single-block case.
        // z starts at 0, local_update returns [3, 4], so z_new = [3, 4], z_diff = [3, 4].
        // Expected dual_res = rho * sqrt(9+16) = 1.0 * 5.0 = 5.0.
        let config = ADMMConfig {
            rho: 1.0,
            adaptive_rho: false,
            ..ADMMConfig::default()
        };
        let mut admm = ConsensusADMM::new(1, 2, config);
        let local_val = Array1::from_vec(vec![3.0f32, 4.0]);
        let (_, dual_res) = admm.iterate(|_, _, _, _| local_val.clone());
        assert!(
            (dual_res - 5.0f32).abs() < 1e-4,
            "Single-block dual residual should be rho*||z_diff||=5.0, got {}",
            dual_res
        );
    }

    #[test]
    fn test_admm_dual_residual_multi_block_not_scaled() {
        // Discriminating: with num_blocks=3 the bug multiplies z_diff‖² by 3, giving
        // dual_res = rho * sqrt(3) * ‖z_diff‖ instead of rho * ‖z_diff‖.
        // Setup: all local_update_fn return [1, 0], dual_vars = 0, z_old = 0.
        // → z_new = mean([1,0],[1,0],[1,0]) = [1, 0], z_diff = [1, 0].
        // Expected dual_res = 2.0 * 1.0 = 2.0 (not 2.0 * sqrt(3) ≈ 3.464).
        let config = ADMMConfig {
            rho: 2.0,
            adaptive_rho: false,
            ..ADMMConfig::default()
        };
        let mut admm = ConsensusADMM::new(3, 2, config);
        let local_val = Array1::from_vec(vec![1.0f32, 0.0]);
        let (_, dual_res) = admm.iterate(|_, _, _, _| local_val.clone());
        let expected = 2.0f32;
        assert!(
            (dual_res - expected).abs() < 1e-4,
            "Multi-block dual residual must be rho*||z_diff||={}, got {} (bug: sqrt(3)*rho*||z_diff||≈{})",
            expected,
            dual_res,
            expected * 3.0f32.sqrt()
        );
    }
}
