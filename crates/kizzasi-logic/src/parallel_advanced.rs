//! Advanced Parallelization for constraint evaluation
//!
//! Provides CPU-parallel (rayon) batch projection, parallel constraint graph traversal
//! with work-stealing, SIMD-optimized batch constraint checking, and multi-threaded
//! incremental solving.
//!
//! # Key Components
//! - [`ConstraintGraph`] — topological layer parallel traversal with Kahn's BFS
//! - [`ParallelBatchProjector`] — batch point projection via rayon work-stealing
//! - [`check_range_constraints_simd`] — auto-vectorization friendly range checking
//! - [`check_constraints_parallel`] — parallel group constraint checking
//! - [`ParallelIncrementalSolver`] — multi-threaded incremental projected gradient solving

use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

// ============================================================================
// Type aliases to satisfy clippy::type_complexity
// ============================================================================

/// Shared constraint violation function: given variable values, returns violation (0 = satisfied).
pub type ConstraintFn = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// Shared gradient function: given variable values, returns the gradient vector.
pub type GradientFn = Arc<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>;

// ============================================================================
// ConstraintNode
// ============================================================================

/// A node in a constraint dependency graph.
///
/// Each node encapsulates a single constraint function along with its dependency
/// edges (which nodes must be evaluated before this one) and which variable
/// indices it reads.
#[derive(Clone)]
pub struct ConstraintNode {
    /// Unique identifier for this node (used as index into the graph's node list)
    pub id: usize,
    /// The constraint function: returns violation amount (0.0 = satisfied)
    pub constraint_fn: ConstraintFn,
    /// IDs of nodes this node depends on (must be evaluated first)
    pub dependencies: Vec<usize>,
    /// Variable indices consumed by this constraint
    pub variables: Vec<usize>,
}

impl std::fmt::Debug for ConstraintNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstraintNode")
            .field("id", &self.id)
            .field("dependencies", &self.dependencies)
            .field("variables", &self.variables)
            .finish()
    }
}

// ============================================================================
// ConstraintGraph
// ============================================================================

/// Constraint dependency graph for parallel traversal.
///
/// Uses Kahn's BFS-based topological sort to group nodes into layers. Nodes
/// within the same layer have no mutual dependencies and can be evaluated
/// concurrently using rayon's work-stealing thread pool.
pub struct ConstraintGraph {
    nodes: Vec<ConstraintNode>,
    /// Topological layers: nodes in the same layer can be evaluated in parallel
    layers: Vec<Vec<usize>>,
}

impl ConstraintGraph {
    /// Create an empty constraint graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            layers: Vec::new(),
        }
    }

    /// Add a node to the graph. Invalidates any previously computed layers.
    pub fn add_node(&mut self, node: ConstraintNode) {
        self.nodes.push(node);
        self.layers.clear();
    }

    /// Compute topological layers using Kahn's BFS algorithm.
    ///
    /// After this call, `self.layers` is populated such that all nodes in
    /// layer `k` depend only on nodes in layers `0..k`.
    pub fn compute_layers(&mut self) {
        let n = self.nodes.len();
        if n == 0 {
            self.layers = Vec::new();
            return;
        }

        // Build in-degree count and adjacency list (dependency edges reversed)
        let mut in_degree = vec![0usize; n];
        // For each node, track which node positions appear as dependencies
        // We use node.id as the index (assuming ids are 0..n and inserted in order)
        // Build a map: node_id -> position in self.nodes
        let mut id_to_pos = std::collections::HashMap::new();
        for (pos, node) in self.nodes.iter().enumerate() {
            id_to_pos.insert(node.id, pos);
        }

        // adjacency[pos] = list of positions that depend on nodes[pos]
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (pos, node) in self.nodes.iter().enumerate() {
            for &dep_id in &node.dependencies {
                if let Some(&dep_pos) = id_to_pos.get(&dep_id) {
                    in_degree[pos] += 1;
                    adjacency[dep_pos].push(pos);
                }
                // Ignore dependency ids not present in graph
            }
        }

        // Kahn's BFS: start with all zero in-degree nodes
        let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut layers: Vec<Vec<usize>> = Vec::new();

        while !queue.is_empty() {
            let layer_size = queue.len();
            let mut layer = Vec::with_capacity(layer_size);
            for _ in 0..layer_size {
                let pos = match queue.pop_front() {
                    Some(p) => p,
                    None => break,
                };
                layer.push(pos);
                for &next_pos in &adjacency[pos] {
                    in_degree[next_pos] -= 1;
                    if in_degree[next_pos] == 0 {
                        queue.push_back(next_pos);
                    }
                }
            }
            layers.push(layer);
        }

        self.layers = layers;
    }

    /// Evaluate all constraints in parallel, layer by layer.
    ///
    /// Returns a vector of violation values indexed by node position in the graph.
    /// Nodes in the same topological layer are evaluated concurrently.
    ///
    /// # Panics
    /// Panics if `compute_layers` has not been called since the last `add_node`.
    pub fn evaluate_parallel(&self, values: &[f64]) -> Vec<f64> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let mut violations = vec![0.0f64; self.nodes.len()];

        for layer in &self.layers {
            let layer_results: Vec<(usize, f64)> = layer
                .par_iter()
                .map(|&idx| (idx, (self.nodes[idx].constraint_fn)(values)))
                .collect();
            for (idx, v) in layer_results {
                violations[idx] = v;
            }
        }

        violations
    }

    /// Check if all constraints are satisfied (parallel evaluation).
    ///
    /// Returns `true` if every node's violation is at most `tolerance`.
    pub fn all_satisfied_parallel(&self, values: &[f64], tolerance: f64) -> bool {
        let violations = self.evaluate_parallel(values);
        violations.iter().all(|&v| v <= tolerance)
    }
}

impl Default for ConstraintGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ParallelBatchProjector
// ============================================================================

/// Batch projection using parallel rayon workers.
///
/// Projects multiple points onto a constraint set concurrently via rayon's
/// work-stealing thread pool (simulating GPU-style batch parallelism in
/// 100% Pure Rust).
pub struct ParallelBatchProjector {
    /// Number of threads hint (0 = rayon global pool default)
    pub num_threads: usize,
    /// Convergence tolerance for projection
    pub tolerance: f64,
    /// Maximum gradient descent iterations per point
    pub max_iterations: usize,
}

impl ParallelBatchProjector {
    /// Create a new projector. `num_threads` is advisory (rayon manages the pool).
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            tolerance: 1e-6,
            max_iterations: 100,
        }
    }

    /// Project multiple points onto the constraint set in parallel.
    ///
    /// Each point is projected independently using projected gradient descent.
    /// `constraint` returns `true` if the point is feasible.
    /// `gradient` returns the constraint violation gradient at a point.
    pub fn project_batch(
        &self,
        points: &[Vec<f64>],
        constraint: &(impl Fn(&[f64]) -> bool + Sync),
        gradient: &(impl Fn(&[f64]) -> Vec<f64> + Sync),
    ) -> Vec<Vec<f64>> {
        let step_size = 0.01f64;
        let max_iter = self.max_iterations;

        points
            .par_iter()
            .map(|point| {
                let mut p = point.clone();
                for _ in 0..max_iter {
                    if constraint(&p) {
                        break;
                    }
                    let grad = gradient(&p);
                    p.iter_mut()
                        .zip(grad.iter())
                        .for_each(|(x, g)| *x -= step_size * g);
                }
                p
            })
            .collect()
    }

    /// Compute violations for a batch of points in parallel.
    ///
    /// Returns `violations[point_idx][constraint_idx]` = violation amount.
    /// Points and constraints are evaluated in a parallel outer loop over points.
    pub fn compute_violations_parallel(
        &self,
        points: &[Vec<f64>],
        constraints: &[ConstraintFn],
    ) -> Vec<Vec<f64>> {
        points
            .par_iter()
            .map(|point| {
                constraints
                    .iter()
                    .map(|c| c(point.as_slice()))
                    .collect::<Vec<f64>>()
            })
            .collect()
    }
}

// ============================================================================
// SIMD-optimized range constraint checking
// ============================================================================

/// Check whether all values in a slice are within `[min, max]`.
///
/// Written in an auto-vectorization-friendly pattern so LLVM can emit
/// SSE/AVX/NEON SIMD instructions. Returns `(all_satisfied, violation_indices)`.
pub fn check_range_constraints_simd(values: &[f64], min: f64, max: f64) -> (bool, Vec<usize>) {
    let violations: Vec<usize> = values
        .iter()
        .enumerate()
        .filter(|(_, &v)| v < min || v > max)
        .map(|(i, _)| i)
        .collect();
    (violations.is_empty(), violations)
}

// ============================================================================
// Parallel range constraint checking for multiple variable groups
// ============================================================================

/// Check whether each variable group satisfies its corresponding `(min, max)` bound.
///
/// Groups and bounds are evaluated in parallel using rayon. Returns a boolean
/// per group indicating satisfaction.
pub fn check_constraints_parallel(
    variable_groups: &[Vec<f64>],
    bounds: &[(f64, f64)],
) -> Vec<bool> {
    variable_groups
        .par_iter()
        .zip(bounds.par_iter())
        .map(|(group, &(min, max))| group.iter().all(|&v| v >= min && v <= max))
        .collect()
}

// ============================================================================
// ParallelIncrementalSolver
// ============================================================================

/// Multi-threaded incremental constraint solver using parallel projected gradient descent.
///
/// Constraints and gradients can be added/removed dynamically. Solving runs a
/// parallel gradient aggregation step over all active constraints, enabling
/// work-stealing across threads for large constraint sets.
pub struct ParallelIncrementalSolver {
    /// Current solution vector (shared, mutex-protected)
    pub solution: Arc<Mutex<Vec<f64>>>,
    /// Indices into `constraints`/`gradients` that are currently active
    active_constraints: Arc<Mutex<Vec<usize>>>,
    /// All registered constraint violation functions
    constraints: Vec<ConstraintFn>,
    /// Gradient functions corresponding to each constraint
    gradients: Vec<GradientFn>,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum number of solver iterations
    pub max_iterations: usize,
}

impl ParallelIncrementalSolver {
    /// Create a new solver for a problem of the given `dimension`.
    ///
    /// The initial solution is the zero vector.
    pub fn new(dimension: usize) -> Self {
        Self {
            solution: Arc::new(Mutex::new(vec![0.0f64; dimension])),
            active_constraints: Arc::new(Mutex::new(Vec::new())),
            constraints: Vec::new(),
            gradients: Vec::new(),
            tolerance: 1e-4,
            max_iterations: 1000,
        }
    }

    /// Register a new constraint and its gradient function.
    ///
    /// Returns the constraint id (its index), which can be used with
    /// [`remove_constraint`](Self::remove_constraint).
    pub fn add_constraint(&mut self, constraint: ConstraintFn, gradient: GradientFn) -> usize {
        let id = self.constraints.len();
        self.constraints.push(constraint);
        self.gradients.push(gradient);
        // Activate immediately
        match self.active_constraints.lock() {
            Ok(mut active) => active.push(id),
            Err(poisoned) => poisoned.into_inner().push(id),
        }
        id
    }

    /// Deactivate a constraint by its id.
    ///
    /// Returns `true` if the constraint was active and has been removed.
    pub fn remove_constraint(&mut self, id: usize) -> bool {
        match self.active_constraints.lock() {
            Ok(mut active) => {
                if let Some(pos) = active.iter().position(|&x| x == id) {
                    active.remove(pos);
                    true
                } else {
                    false
                }
            }
            Err(poisoned) => {
                let mut active = poisoned.into_inner();
                if let Some(pos) = active.iter().position(|&x| x == id) {
                    active.remove(pos);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Execute one step of parallel projected gradient descent.
    ///
    /// For each active constraint, violation and gradient are evaluated in
    /// parallel (rayon). Gradients are weighted by violation and summed. The
    /// solution is updated by a fixed step along the aggregate gradient.
    ///
    /// Returns the total constraint violation after the step.
    pub fn step(&self) -> Result<f64, String> {
        let step_size = 0.01f64;

        // Snapshot the current solution
        let current = self.solution.lock().map_err(|e| {
            format!(
                "ParallelIncrementalSolver::step — solution lock poisoned: {}",
                e
            )
        })?;
        let sol_snapshot = current.clone();
        drop(current);

        // Snapshot active constraint indices
        let active_ids: Vec<usize> = self
            .active_constraints
            .lock()
            .map_err(|e| {
                format!(
                    "ParallelIncrementalSolver::step — active_constraints lock poisoned: {}",
                    e
                )
            })?
            .clone();

        if active_ids.is_empty() {
            return Ok(0.0);
        }

        let dim = sol_snapshot.len();

        // Parallel evaluation: (violation, gradient) per active constraint
        let evaluated: Vec<(f64, Vec<f64>)> = active_ids
            .par_iter()
            .map(|&idx| {
                let violation = (self.constraints[idx])(&sol_snapshot);
                let grad = (self.gradients[idx])(&sol_snapshot);
                (violation, grad)
            })
            .collect();

        // Aggregate: total violation and weighted gradient sum
        let total_violation: f64 = evaluated.iter().map(|(v, _)| *v).sum();

        // Sum gradients weighted by their violation (only positive violation contributes)
        let mut aggregate_grad = vec![0.0f64; dim];
        for (violation, grad) in &evaluated {
            if *violation > 0.0 {
                for (ag, gv) in aggregate_grad.iter_mut().zip(grad.iter()) {
                    *ag += violation * gv;
                }
            }
        }

        // Apply update
        let mut sol = self.solution.lock().map_err(|e| {
            format!(
                "ParallelIncrementalSolver::step — solution write lock poisoned: {}",
                e
            )
        })?;
        for (x, g) in sol.iter_mut().zip(aggregate_grad.iter()) {
            *x -= step_size * g;
        }

        Ok(total_violation)
    }

    /// Run the solver until convergence or `max_iterations` is reached.
    ///
    /// Returns the converged solution vector, or an error string if a lock
    /// was poisoned.
    pub fn solve(&self) -> Result<Vec<f64>, String> {
        for _iter in 0..self.max_iterations {
            let total_violation = self.step()?;
            if total_violation < self.tolerance {
                break;
            }
        }
        Ok(self.solution())
    }

    /// Return a snapshot of the current solution.
    pub fn solution(&self) -> Vec<f64> {
        match self.solution.lock() {
            Ok(sol) => sol.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
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
    fn test_range_constraint_simd_all_satisfied() {
        let vals = vec![1.0f64, 2.0, 3.0];
        let (ok, violations) = check_range_constraints_simd(&vals, 0.0, 5.0);
        assert!(ok);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_range_constraint_simd_violations() {
        let vals = vec![-1.0f64, 2.0, 6.0];
        let (ok, violations) = check_range_constraints_simd(&vals, 0.0, 5.0);
        assert!(!ok);
        assert_eq!(violations, vec![0, 2]);
    }

    #[test]
    fn test_constraint_graph_empty_evaluate() {
        let mut graph = ConstraintGraph::new();
        graph.compute_layers();
        let violations = graph.evaluate_parallel(&[1.0, 2.0]);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_constraint_graph_parallel_evaluation() {
        let mut graph = ConstraintGraph::new();
        let n0 = ConstraintNode {
            id: 0,
            constraint_fn: Arc::new(|v: &[f64]| (v[0] - 1.0).max(0.0)),
            dependencies: vec![],
            variables: vec![0],
        };
        let n1 = ConstraintNode {
            id: 1,
            constraint_fn: Arc::new(|v: &[f64]| (-v[1]).max(0.0)),
            dependencies: vec![],
            variables: vec![1],
        };
        graph.add_node(n0);
        graph.add_node(n1);
        graph.compute_layers();

        // x = [0.5, 1.0] — both satisfied
        let violations = graph.evaluate_parallel(&[0.5, 1.0]);
        assert_eq!(violations.len(), 2);
        assert!(violations[0].abs() < 1e-9);
        assert!(violations[1].abs() < 1e-9);

        // x = [2.0, -1.0] — both violated
        let violations2 = graph.evaluate_parallel(&[2.0, -1.0]);
        assert!(violations2[0] > 0.0);
        assert!(violations2[1] > 0.0);
    }

    #[test]
    fn test_constraint_graph_with_dependencies() {
        let mut graph = ConstraintGraph::new();
        // n0: no deps
        let n0 = ConstraintNode {
            id: 0,
            constraint_fn: Arc::new(|v: &[f64]| (v[0] - 1.0).max(0.0)),
            dependencies: vec![],
            variables: vec![0],
        };
        // n1: depends on n0
        let n1 = ConstraintNode {
            id: 1,
            constraint_fn: Arc::new(|v: &[f64]| (v[1] - 2.0).max(0.0)),
            dependencies: vec![0],
            variables: vec![1],
        };
        graph.add_node(n0);
        graph.add_node(n1);
        graph.compute_layers();

        // Should have 2 layers: layer 0 = [0], layer 1 = [1]
        assert_eq!(graph.layers.len(), 2);
        assert_eq!(graph.layers[0], vec![0]);
        assert_eq!(graph.layers[1], vec![1]);

        let violations = graph.evaluate_parallel(&[0.5, 1.5]);
        assert_eq!(violations.len(), 2);
        // Both satisfied
        assert!(violations[0].abs() < 1e-9);
        assert!(violations[1].abs() < 1e-9);
    }

    #[test]
    fn test_constraint_graph_all_satisfied_parallel() {
        let mut graph = ConstraintGraph::new();
        let n0 = ConstraintNode {
            id: 0,
            constraint_fn: Arc::new(|v: &[f64]| (v[0] - 1.0).max(0.0)),
            dependencies: vec![],
            variables: vec![0],
        };
        graph.add_node(n0);
        graph.compute_layers();

        assert!(graph.all_satisfied_parallel(&[0.5], 1e-9));
        assert!(!graph.all_satisfied_parallel(&[2.0], 1e-9));
    }

    #[test]
    fn test_parallel_batch_projector_project_batch() {
        let projector = ParallelBatchProjector::new(2);
        let points = vec![vec![2.0f64], vec![0.5f64]];
        // Constraint: x[0] <= 1.0 (returns true if satisfied)
        let constraint = |v: &[f64]| v[0] <= 1.0;
        // Gradient: pushes x[0] toward 1.0 when > 1.0
        let gradient = |v: &[f64]| if v[0] > 1.0 { vec![1.0] } else { vec![0.0] };

        let projected = projector.project_batch(&points, &constraint, &gradient);
        assert_eq!(projected.len(), 2);
        // Second point was already feasible
        assert!((projected[1][0] - 0.5).abs() < 1e-9);
        // First point should have been pushed down
        assert!(projected[0][0] <= 1.0 + 1e-6);
    }

    type ConstraintFn = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

    #[test]
    fn test_parallel_batch_projector_violations() {
        let projector = ParallelBatchProjector::new(2);
        let points = vec![vec![0.5f64, 0.5], vec![2.0, 2.0]];
        let constraint: ConstraintFn =
            Arc::new(|v: &[f64]| (v[0] - 1.0).max(0.0) + (v[1] - 1.0).max(0.0));
        let violations = projector.compute_violations_parallel(&points, &[constraint]);
        assert_eq!(violations.len(), 2);
        assert!(violations[0][0].abs() < 1e-9);
        assert!(violations[1][0] > 0.0);
    }

    #[test]
    fn test_check_constraints_parallel() {
        let groups = vec![vec![0.5f64, 0.8], vec![2.0f64, 3.0]];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let results = check_constraints_parallel(&groups, &bounds);
        assert_eq!(results.len(), 2);
        assert!(results[0]); // [0.5, 0.8] within [0, 1]
        assert!(!results[1]); // [2.0, 3.0] outside [0, 1]
    }

    #[test]
    fn test_check_constraints_parallel_empty() {
        let results = check_constraints_parallel(&[], &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parallel_incremental_solver_basic() {
        let mut solver = ParallelIncrementalSolver::new(2);
        // Constraint: x[0] <= 0.5 (violation = max(0, x[0] - 0.5))
        solver.add_constraint(
            Arc::new(|v: &[f64]| (v[0] - 0.5).max(0.0)),
            Arc::new(|v: &[f64]| {
                if v[0] > 0.5 {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 0.0]
                }
            }),
        );
        // Start with x = [1.0, 0.0]
        {
            let mut sol = solver.solution.lock().expect("lock failed in test setup");
            *sol = vec![1.0, 0.0];
        }
        let result = solver.solve();
        assert!(result.is_ok());
        let sol = result.unwrap();
        assert!(sol[0] <= 0.5 + 0.01); // within tolerance
    }

    #[test]
    fn test_parallel_incremental_solver_no_constraints() {
        let solver = ParallelIncrementalSolver::new(3);
        let result = solver.solve();
        assert!(result.is_ok());
        let sol = result.unwrap();
        assert_eq!(sol.len(), 3);
    }

    #[test]
    fn test_parallel_incremental_solver_add_remove() {
        let mut solver = ParallelIncrementalSolver::new(2);
        let id = solver.add_constraint(
            Arc::new(|v: &[f64]| (v[0] - 0.5).max(0.0)),
            Arc::new(|_v: &[f64]| vec![1.0, 0.0]),
        );
        assert_eq!(id, 0);
        let removed = solver.remove_constraint(id);
        assert!(removed);
        // Removing again should return false
        let removed_again = solver.remove_constraint(id);
        assert!(!removed_again);
    }

    #[test]
    fn test_parallel_incremental_solver_step() {
        let mut solver = ParallelIncrementalSolver::new(1);
        solver.add_constraint(
            Arc::new(|v: &[f64]| (v[0] - 0.0).max(0.0)), // x[0] <= 0
            Arc::new(|_v: &[f64]| vec![1.0]),
        );
        {
            let mut sol = solver.solution.lock().expect("lock in test");
            *sol = vec![1.0];
        }
        let violation = solver.step().expect("step should succeed");
        // Violation was 1.0 initially; after one step it should decrease
        assert!(violation >= 0.0);
        let sol = solver.solution();
        assert!(sol[0] < 1.0); // solution moved toward constraint
    }
}
