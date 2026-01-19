//! Quadratic Programming (QP) solver integration
//!
//! This module provides integration with the OSQP solver for solving
//! constrained quadratic programming problems of the form:
//!
//! minimize    (1/2) x^T P x + q^T x
//! subject to  l <= A x <= u
//!
//! This is useful for projecting points onto the intersection of
//! multiple quadratic and linear constraints.

use crate::error::{LogicError, LogicResult};
use scirs2_core::ndarray::{Array1, Array2};

#[cfg(feature = "qp-solver")]
use osqp::{CscMatrix, Problem, Settings};

/// QP solver wrapper for constraint projection
///
/// Solves quadratic programming problems to find the closest point
/// satisfying a set of linear and quadratic constraints.
pub struct QPSolver {
    /// Convergence tolerance
    tolerance: f32,
    /// Maximum number of iterations
    max_iter: usize,
}

impl Default for QPSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl QPSolver {
    /// Create a new QP solver with default settings
    pub fn new() -> Self {
        Self {
            tolerance: 1e-6,
            max_iter: 10000,
        }
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Project a point onto the intersection of linear constraints
    ///
    /// Solves:
    /// minimize    ||x - x0||^2
    /// subject to  A x <= b
    ///
    /// # Arguments
    /// * `x0` - Initial point to project
    /// * `a_matrices` - List of constraint coefficient matrices (each row is a constraint)
    /// * `b_vectors` - List of constraint bounds
    ///
    /// # Returns
    /// The projected point satisfying all constraints
    #[cfg(feature = "qp-solver")]
    pub fn project_linear(
        &self,
        x0: &Array1<f32>,
        a_matrices: Vec<Array2<f32>>,
        b_vectors: Vec<Array1<f32>>,
    ) -> LogicResult<Array1<f32>> {
        let n = x0.len();

        // Build P matrix (identity for ||x - x0||^2)
        let p_data: Vec<f64> = (0..n).map(|_| 2.0).collect();
        let p_indices: Vec<usize> = (0..n).collect();
        let p_indptr: Vec<usize> = (0..=n).collect();

        let p = CscMatrix {
            nrows: n,
            ncols: n,
            indptr: p_indptr.into(),
            indices: p_indices.into(),
            data: p_data.into(),
        };

        // Build q vector (-2 * x0 for minimizing ||x - x0||^2)
        let q: Vec<f64> = x0.iter().map(|&x| -2.0 * x as f64).collect();

        // Build constraint matrices A and bounds l, u
        let mut a_rows = Vec::new();
        let mut a_cols = Vec::new();
        let mut a_data = Vec::new();
        let mut lower = Vec::new();
        let mut upper = Vec::new();

        let mut row_idx = 0;
        for (a_mat, b_vec) in a_matrices.iter().zip(b_vectors.iter()) {
            let (m, n_cols) = a_mat.dim();
            assert_eq!(n_cols, n, "Constraint matrix dimension mismatch");
            assert_eq!(b_vec.len(), m, "Constraint vector dimension mismatch");

            for i in 0..m {
                for j in 0..n {
                    let val = a_mat[[i, j]];
                    if val.abs() > 1e-10 {
                        a_rows.push(row_idx);
                        a_cols.push(j);
                        a_data.push(val as f64);
                    }
                }
                lower.push(f64::NEG_INFINITY);
                upper.push(b_vec[i] as f64);
                row_idx += 1;
            }
        }

        // Convert to CSC format
        let a = Self::build_csc_matrix(row_idx, n, &a_rows, &a_cols, &a_data)?;

        // Setup and solve
        let settings = Settings::default()
            .eps_abs(self.tolerance as f64)
            .eps_rel(self.tolerance as f64)
            .max_iter(self.max_iter as u32)
            .verbose(false);

        let mut problem = Problem::new(p, &q, a, &lower, &upper, &settings).map_err(|e| {
            LogicError::ProjectionFailed(format!("Failed to create QP problem: {:?}", e))
        })?;

        let status = problem.solve();

        // Extract solution from status
        // Accept both Solved and MaxIterationsReached if residuals are small
        let solution_data = match status {
            osqp::Status::Solved(ref sol) => sol.x(),
            osqp::Status::MaxIterationsReached(ref sol) => {
                // Check if residuals are acceptable
                if sol.pri_res() < 1e-3 && sol.dua_res() < 1e-3 {
                    sol.x()
                } else {
                    return Err(LogicError::ProjectionFailed(format!(
                        "QP solver did not converge: pri_res={}, dua_res={}",
                        sol.pri_res(),
                        sol.dua_res()
                    )));
                }
            }
            _ => {
                return Err(LogicError::ProjectionFailed(format!(
                    "QP solver failed: {:?}",
                    status
                )))
            }
        };

        let solution: Vec<f32> = solution_data.iter().copied().map(|x| x as f32).collect();
        Ok(Array1::from_vec(solution))
    }

    /// Project a point onto the intersection of linear and quadratic constraints
    ///
    /// Solves:
    /// minimize    ||x - x0||^2
    /// subject to  A x <= b
    ///             x^T Q x + c^T x <= r  (for quadratic constraints)
    ///
    /// Note: This requires linearization of quadratic constraints
    #[cfg(feature = "qp-solver")]
    pub fn project_quadratic(
        &self,
        x0: &Array1<f32>,
        linear_a: Vec<Array2<f32>>,
        linear_b: Vec<Array1<f32>>,
        quadratic_q: Vec<Array2<f32>>,
        quadratic_c: Vec<Array1<f32>>,
        quadratic_r: Vec<f32>,
    ) -> LogicResult<Array1<f32>> {
        // For now, use sequential quadratic programming (SQP) approximation
        // Linearize quadratic constraints around current point
        let mut current = x0.clone();

        for _iter in 0..10 {
            let mut all_a = linear_a.clone();
            let mut all_b = linear_b.clone();

            // Linearize each quadratic constraint
            for ((q, c), r) in quadratic_q
                .iter()
                .zip(quadratic_c.iter())
                .zip(quadratic_r.iter())
            {
                // Gradient at current point: 2 Q x + c
                let grad: Array1<f32> = 2.0 * q.dot(&current) + c.clone();

                // Linear approximation: grad^T (x - current) + current^T Q current + c^T current <= r
                // Simplifies to: grad^T x <= r - current^T Q current - c^T current + grad^T current
                let qx = q.dot(&current);
                let val = current.dot(&qx) + c.dot(&current);
                let rhs = r - val + grad.dot(&current);

                // Add to constraints
                let a_row =
                    Array2::from_shape_vec((1, current.len()), grad.to_vec()).map_err(|e| {
                        LogicError::InvalidConstraint(format!(
                            "Failed to create constraint matrix: {}",
                            e
                        ))
                    })?;
                all_a.push(a_row);
                all_b.push(Array1::from_vec(vec![rhs]));
            }

            // Solve linearized problem
            let next = self.project_linear(&current, all_a, all_b)?;

            // Check convergence
            let diff: f32 = next
                .iter()
                .zip(current.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();

            if diff < self.tolerance {
                return Ok(next);
            }

            current = next;
        }

        Ok(current)
    }

    #[cfg(feature = "qp-solver")]
    fn build_csc_matrix(
        nrows: usize,
        ncols: usize,
        rows: &[usize],
        cols: &[usize],
        data: &[f64],
    ) -> LogicResult<CscMatrix<'static>> {
        // Convert COO format to CSC format
        let nnz = data.len();
        if rows.len() != nnz || cols.len() != nnz {
            return Err(LogicError::InvalidConstraint(
                "Matrix dimension mismatch".into(),
            ));
        }

        // Create triplets and sort by column then row
        let mut triplets: Vec<(usize, usize, f64)> = rows
            .iter()
            .zip(cols.iter())
            .zip(data.iter())
            .map(|((r, c), d)| (*c, *r, *d))
            .collect();
        triplets.sort_by_key(|t| (t.0, t.1));

        let mut indices = Vec::with_capacity(nnz);
        let mut data_vec = Vec::with_capacity(nnz);
        let mut indptr = vec![0];

        let mut current_col = 0;
        for (col, row, val) in triplets {
            while current_col < col {
                indptr.push(indices.len());
                current_col += 1;
            }
            indices.push(row);
            data_vec.push(val);
        }

        while current_col < ncols {
            indptr.push(indices.len());
            current_col += 1;
        }

        Ok(CscMatrix {
            nrows,
            ncols,
            indptr: indptr.into(),
            indices: indices.into(),
            data: data_vec.into(),
        })
    }
}

#[cfg(not(feature = "qp-solver"))]
impl QPSolver {
    /// Fallback implementation when QP solver is not enabled
    pub fn project_linear(
        &self,
        _x0: &Array1<f32>,
        _a_matrices: Vec<Array2<f32>>,
        _b_vectors: Vec<Array1<f32>>,
    ) -> LogicResult<Array1<f32>> {
        Err(LogicError::ProjectionFailed(
            "QP solver feature not enabled. Enable with --features qp-solver".into(),
        ))
    }

    /// Fallback implementation when QP solver is not enabled
    pub fn project_quadratic(
        &self,
        _x0: &Array1<f32>,
        _linear_a: Vec<Array2<f32>>,
        _linear_b: Vec<Array1<f32>>,
        _quadratic_q: Vec<Array2<f32>>,
        _quadratic_c: Vec<Array1<f32>>,
        _quadratic_r: Vec<f32>,
    ) -> LogicResult<Array1<f32>> {
        Err(LogicError::ProjectionFailed(
            "QP solver feature not enabled. Enable with --features qp-solver".into(),
        ))
    }
}

#[cfg(all(test, feature = "qp-solver"))]
mod tests {
    use super::*;

    #[test]
    fn test_qp_solver_simple_projection() {
        let solver = QPSolver::new();

        // Project point (2, 2) onto x + y <= 1
        let x0 = Array1::from_vec(vec![2.0, 2.0]);
        let a = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();
        let b = Array1::from_vec(vec![1.0]);

        let result = solver.project_linear(&x0, vec![a], vec![b]);
        match &result {
            Ok(_) => {}
            Err(e) => eprintln!("QP solver error: {:?}", e),
        }
        assert!(result.is_ok(), "QP solver failed: {:?}", result.err());

        let projected = result.unwrap();
        // Sum should be <= 1
        let sum: f32 = projected.iter().sum();
        assert!(sum <= 1.0 + 1e-3);
    }

    #[test]
    fn test_qp_solver_box_constraint() {
        let solver = QPSolver::new();

        // Project (2, -1) onto [0, 1] x [0, 1]
        let x0 = Array1::from_vec(vec![2.0, -1.0]);

        // x <= 1, y <= 1, -x <= 0, -y <= 0
        let a =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0]).unwrap();
        let b = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]);

        let result = solver.project_linear(&x0, vec![a], vec![b]);
        assert!(result.is_ok());

        let projected = result.unwrap();
        assert!(projected[0] >= -1e-3 && projected[0] <= 1.0 + 1e-3);
        assert!(projected[1] >= -1e-3 && projected[1] <= 1.0 + 1e-3);
    }
}
