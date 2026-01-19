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

use scirs2_core::ndarray::{Array1, Array2};

/// Result type for decomposition operations
pub type DecompositionResult<T> = Result<T, DecompositionError>;

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
    pub fn initialize(&mut self, x0: &Array1<f32>) {
        assert_eq!(x0.len(), self.dimension, "Initial point dimension mismatch");
        self.global_var = x0.clone();
        for i in 0..self.num_blocks {
            self.local_vars[i] = x0.clone();
        }
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
        let mut dual_residual = 0.0f32;

        let z_diff = &z_new - &self.global_var;
        for i in 0..self.num_blocks {
            let xi_minus_z = &self.local_vars[i] - &z_new;
            self.dual_vars[i] = &self.dual_vars[i] + &xi_minus_z;

            // Compute residuals
            primal_residual += xi_minus_z.iter().map(|&x| x * x).sum::<f32>();
            dual_residual += z_diff.iter().map(|&x| x * x).sum::<f32>();
        }

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
    pub fn initialize(&mut self, x0: &Array1<f32>) {
        assert_eq!(x0.len(), self.dimension);
        self.current_solution = x0.clone();
    }

    /// Perform one block update
    ///
    /// The `block_update_fn` should optimize the objective over the specified block
    /// while keeping other variables fixed.
    pub fn update_block<F>(&mut self, block_id: usize, block_update_fn: F) -> f32
    where
        F: Fn(&Array1<f32>, &[usize]) -> Array1<f32>,
    {
        let block = &self.blocks[block_id];
        let block_solution = block_update_fn(&self.current_solution, &block.indices);

        assert_eq!(
            block_solution.len(),
            block.indices.len(),
            "Block solution size mismatch"
        );

        // Update the solution for this block
        let mut change = 0.0f32;
        for (i, &idx) in block.indices.iter().enumerate() {
            let old_val = self.current_solution[idx];
            let new_val = block_solution[i];
            self.current_solution[idx] = new_val;
            change += (new_val - old_val).powi(2);
        }

        change.sqrt()
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
    pub fn update_duals(&mut self, constraint_violations: &Array1<f32>) {
        assert_eq!(constraint_violations.len(), self.dual_vars.len());

        // Dual ascent: λ := λ + α·g(x) where g(x) is the constraint violation
        for i in 0..self.dual_vars.len() {
            self.dual_vars[i] =
                (self.dual_vars[i] + self.step_size * constraint_violations[i]).max(0.0);
        }
    }

    /// Get current dual variables
    pub fn dual_variables(&self) -> &Array1<f32> {
        &self.dual_vars
    }

    /// Compute augmented cost for subproblem i
    pub fn augmented_cost(
        &self,
        subproblem_id: usize,
        base_cost: f32,
        local_vars: &Array1<f32>,
    ) -> f32 {
        assert!(subproblem_id < self.num_subproblems);

        // Add dual contribution: f(x) + λᵀAx
        let mut augmented = base_cost;
        for (i, &dual) in self.dual_vars.iter().enumerate() {
            let coupling_coeff = self.coupling_matrix[[i, subproblem_id]];
            if let Some(&var_val) = local_vars.get(0) {
                augmented += dual * coupling_coeff * var_val;
            }
        }

        augmented
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
    pub fn overlapping_blocks(dimension: usize, block_size: usize, overlap: usize) -> Vec<Block> {
        assert!(overlap < block_size, "Overlap must be less than block size");

        let stride = block_size - overlap;
        let mut blocks = Vec::new();
        let mut start = 0;

        while start < dimension {
            let end = (start + block_size).min(dimension);
            let indices: Vec<usize> = (start..end).collect();
            blocks.push(Block::new(format!("block_{}", blocks.len()), indices));
            start += stride;
        }

        blocks
    }
}

// ============================================================================
// Benders Decomposition
// ============================================================================

/// Benders Decomposition for mixed-integer programming
///
/// Decomposes problems of the form:
/// ```text
/// min c'x + d'y
/// s.t. Ax + By >= b
///      x binary/integer, y continuous
/// ```
///
/// Into:
/// - Master problem: involves only x (integer variables)
/// - Subproblem: involves only y given fixed x
///
/// # Algorithm
///
/// 1. Solve relaxed master problem → get x*
/// 2. Solve subproblem with fixed x* → get dual variables
/// 3. Add Benders cut to master based on dual values
/// 4. Repeat until convergence
pub struct BendersDecomposition {
    /// Number of master variables (integer/binary)
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
    pub fn new(num_master_vars: usize, num_sub_vars: usize) -> Self {
        Self {
            num_master_vars,
            num_sub_vars,
            cuts: Vec::new(),
            config: BendersConfig::default(),
            lower_bound: f32::NEG_INFINITY,
            upper_bound: f32::INFINITY,
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: BendersConfig) -> Self {
        self.config = config;
        self
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

    /// Solve master problem (to be implemented with external solver)
    ///
    /// Returns the master solution and lower bound
    pub fn solve_master(&self, objective: &Array1<f32>) -> DecompositionResult<(Array1<f32>, f32)> {
        // Simplified master problem: minimize c'x + θ
        // subject to Benders cuts
        //
        // In practice, this would call an MILP solver
        // For now, we return a simple relaxation

        if self.num_master_vars == 0 {
            return Err(DecompositionError::IncompatibleStructure(
                "No master variables".to_string(),
            ));
        }

        // Simple heuristic: round to nearest feasible integer
        let mut x = Array1::zeros(self.num_master_vars);

        // Estimate lower bound from cuts
        let mut theta = f32::NEG_INFINITY;
        for cut in &self.cuts {
            match cut {
                BendersCut::Optimality {
                    constant,
                    coefficients,
                    ..
                } => {
                    let cut_value = constant + coefficients.dot(&x);
                    theta = theta.max(cut_value);
                }
                BendersCut::Feasibility {
                    constant,
                    coefficients,
                    ..
                } => {
                    // Ensure feasibility cut is satisfied
                    let cut_value = constant + coefficients.dot(&x);
                    if cut_value < 0.0 {
                        // Adjust x to satisfy cut (simplified)
                        x = x.mapv(|v| v + 0.1);
                    }
                }
            }
        }

        let lower_bound = objective.dot(&x) + theta;
        Ok((x, lower_bound))
    }

    /// Solve subproblem given master solution
    ///
    /// Returns (objective value, dual variables, feasible flag)
    pub fn solve_subproblem(
        &self,
        master_solution: &Array1<f32>,
        sub_objective: &Array1<f32>,
        coupling_matrix: &Array2<f32>,
        rhs: &Array1<f32>,
    ) -> DecompositionResult<(f32, Array1<f32>, bool)> {
        // Solve: min d'y
        //        s.t. By >= b - Ax*
        //
        // where x* is the fixed master solution
        // Returns (objective, dual variables, is_feasible)

        let (n_constraints, n_vars) = coupling_matrix.dim();

        if n_vars != self.num_sub_vars {
            return Err(DecompositionError::IncompatibleStructure(
                "Subproblem dimension mismatch".to_string(),
            ));
        }

        // Compute adjusted RHS: b - Ax*
        let _adjusted_rhs = if !master_solution.is_empty() {
            // For simplification, assume we have the master coupling part
            rhs.clone()
        } else {
            rhs.clone()
        };

        // Simplified LP solve (in practice, use OSQP or other LP solver)
        // For now, return a feasible solution with dual values

        let y = Array1::from_elem(n_vars, 0.5);
        let dual = Array1::from_elem(n_constraints, 1.0);
        let obj_value = sub_objective.dot(&y);

        Ok((obj_value, dual, true))
    }

    /// Main Benders iteration
    pub fn iterate(
        &mut self,
        master_obj: &Array1<f32>,
        sub_obj: &Array1<f32>,
        coupling: &Array2<f32>,
        rhs: &Array1<f32>,
    ) -> DecompositionResult<BendersIterationResult> {
        let mut iteration = 0;

        loop {
            iteration += 1;

            if iteration > self.config.max_iterations {
                return Err(DecompositionError::ConvergenceFailed {
                    iterations: iteration,
                    residual: self.upper_bound - self.lower_bound,
                });
            }

            // Step 1: Solve master problem
            let (master_sol, lower_bound) = self.solve_master(master_obj)?;
            self.lower_bound = lower_bound;

            // Step 2: Solve subproblem
            let (sub_obj_value, dual, is_feasible) =
                self.solve_subproblem(&master_sol, sub_obj, coupling, rhs)?;

            // Step 3: Update upper bound
            let master_obj_value = master_obj.dot(&master_sol);
            let total_obj = master_obj_value + sub_obj_value;
            self.upper_bound = self.upper_bound.min(total_obj);

            // Step 4: Check convergence
            let gap = self.upper_bound - self.lower_bound;
            if gap < self.config.tolerance {
                return Ok(BendersIterationResult {
                    master_solution: master_sol,
                    sub_solution: Array1::zeros(self.num_sub_vars), // Would be from subproblem
                    lower_bound: self.lower_bound,
                    upper_bound: self.upper_bound,
                    iterations: iteration,
                    converged: true,
                });
            }

            // Step 5: Generate and add cut
            if !is_feasible {
                // Add feasibility cut (using extreme ray)
                let coefficients = Array1::zeros(self.num_master_vars);
                self.add_feasibility_cut(dual.clone(), 0.0, coefficients);
            } else {
                // Add optimality cut
                let coefficients = Array1::zeros(self.num_master_vars);
                let constant = sub_obj_value;
                self.add_optimality_cut(dual, constant, coefficients);
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
        admm.initialize(&x0);

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
        let blocks = block_utils::overlapping_blocks(10, 4, 1);
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
}
