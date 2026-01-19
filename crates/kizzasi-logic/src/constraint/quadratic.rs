use serde::{Deserialize, Serialize};

// ============================================================================
// Quadratic Constraints
// ============================================================================

/// Type of quadratic constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuadraticConstraintType {
    /// x'Qx + c'x <= b
    LessEq,
    /// x'Qx + c'x >= b
    GreaterEq,
    /// |x'Qx + c'x - b| <= tolerance
    Equality { tolerance: f32 },
}

/// Quadratic constraint: x'Qx + c'x (op) b
///
/// Represents a quadratic constraint where:
/// - Q is an n×n symmetric matrix (stored as upper triangular for efficiency)
/// - c is an n-vector (linear term)
/// - b is a scalar (right-hand side)
///
/// Common uses:
/// - Ellipsoid constraints: x'Qx <= 1 (with Q positive definite)
/// - Ball constraints: ||x||² <= r² (Q = I, c = 0, b = r²)
/// - Energy bounds: kinetic + potential <= max
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadraticConstraint {
    /// Quadratic matrix Q (stored as full n×n for simplicity)
    /// Q should be symmetric for meaningful constraints
    q_matrix: Vec<Vec<f32>>,
    /// Linear coefficient vector c
    linear: Vec<f32>,
    /// Right-hand side b
    rhs: f32,
    /// Constraint type
    constraint_type: QuadraticConstraintType,
    /// Weight for loss computation
    weight: f32,
}

impl QuadraticConstraint {
    /// Create a quadratic less-than-or-equal constraint: x'Qx + c'x <= b
    pub fn less_eq(q_matrix: Vec<Vec<f32>>, linear: Vec<f32>, rhs: f32) -> Self {
        Self {
            q_matrix,
            linear,
            rhs,
            constraint_type: QuadraticConstraintType::LessEq,
            weight: 1.0,
        }
    }

    /// Create a quadratic greater-than-or-equal constraint: x'Qx + c'x >= b
    pub fn greater_eq(q_matrix: Vec<Vec<f32>>, linear: Vec<f32>, rhs: f32) -> Self {
        Self {
            q_matrix,
            linear,
            rhs,
            constraint_type: QuadraticConstraintType::GreaterEq,
            weight: 1.0,
        }
    }

    /// Create a quadratic equality constraint: |x'Qx + c'x - b| <= tolerance
    pub fn equality(q_matrix: Vec<Vec<f32>>, linear: Vec<f32>, rhs: f32, tolerance: f32) -> Self {
        Self {
            q_matrix,
            linear,
            rhs,
            constraint_type: QuadraticConstraintType::Equality { tolerance },
            weight: 1.0,
        }
    }

    /// Create a ball constraint: ||x - center||² <= radius²
    ///
    /// Equivalent to: x'Ix - 2*center'x + ||center||² <= radius²
    pub fn ball(center: Vec<f32>, radius: f32) -> Self {
        let n = center.len();
        // Q = I (identity matrix)
        let q_matrix: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut row = vec![0.0; n];
                row[i] = 1.0;
                row
            })
            .collect();
        // c = -2 * center
        let linear: Vec<f32> = center.iter().map(|&ci| -2.0 * ci).collect();
        // b = radius² - ||center||²
        let center_norm_sq: f32 = center.iter().map(|c| c * c).sum();
        let rhs = radius * radius - center_norm_sq;

        Self::less_eq(q_matrix, linear, rhs)
    }

    /// Create an ellipsoid constraint: (x - center)'A(x - center) <= 1
    ///
    /// A should be positive definite. The ellipsoid has semi-axes along eigenvectors
    /// with lengths 1/sqrt(eigenvalues).
    pub fn ellipsoid(a_matrix: Vec<Vec<f32>>, center: Vec<f32>) -> Self {
        let n = center.len();
        // Expand (x - c)'A(x - c) = x'Ax - 2c'Ax + c'Ac
        // Q = A
        let q_matrix = a_matrix.clone();
        // c = -2 * A * center
        let linear: Vec<f32> = (0..n)
            .map(|i| {
                let ac_i: f32 = a_matrix[i]
                    .iter()
                    .zip(center.iter())
                    .map(|(a, c)| a * c)
                    .sum();
                -2.0 * ac_i
            })
            .collect();
        // b = 1 - center'A*center
        let a_center: Vec<f32> = (0..n)
            .map(|i| {
                a_matrix[i]
                    .iter()
                    .zip(center.iter())
                    .map(|(a, c)| a * c)
                    .sum()
            })
            .collect();
        let center_a_center: f32 = center
            .iter()
            .zip(a_center.iter())
            .map(|(c, ac)| c * ac)
            .sum();
        let rhs = 1.0 - center_a_center;

        Self::less_eq(q_matrix, linear, rhs)
    }

    /// Set the constraint weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Compute x'Qx
    fn quadratic_form(&self, x: &[f32]) -> f32 {
        let n = x.len();
        let mut result = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i < self.q_matrix.len() && j < self.q_matrix[i].len() {
                    result += x[i] * self.q_matrix[i][j] * x[j];
                }
            }
        }
        result
    }

    /// Compute c'x
    fn linear_term(&self, x: &[f32]) -> f32 {
        self.linear.iter().zip(x.iter()).map(|(c, xi)| c * xi).sum()
    }

    /// Compute the full constraint value: x'Qx + c'x
    pub fn evaluate(&self, x: &[f32]) -> f32 {
        self.quadratic_form(x) + self.linear_term(x)
    }

    /// Check if values satisfy the constraint
    pub fn check(&self, x: &[f32]) -> bool {
        let val = self.evaluate(x);
        match &self.constraint_type {
            QuadraticConstraintType::LessEq => val <= self.rhs,
            QuadraticConstraintType::GreaterEq => val >= self.rhs,
            QuadraticConstraintType::Equality { tolerance } => (val - self.rhs).abs() <= *tolerance,
        }
    }

    /// Compute violation amount (0 if satisfied)
    pub fn violation(&self, x: &[f32]) -> f32 {
        let val = self.evaluate(x);
        match &self.constraint_type {
            QuadraticConstraintType::LessEq => (val - self.rhs).max(0.0),
            QuadraticConstraintType::GreaterEq => (self.rhs - val).max(0.0),
            QuadraticConstraintType::Equality { tolerance } => {
                let diff = (val - self.rhs).abs();
                (diff - tolerance).max(0.0)
            }
        }
    }

    /// Compute gradient of x'Qx + c'x with respect to x
    ///
    /// For general Q: gradient = (Q + Q')x + c
    /// For symmetric Q: gradient = 2Qx + c
    pub fn gradient(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let m = self.q_matrix.len(); // number of rows in Q
        let mut grad = self.linear.clone();
        grad.resize(n, 0.0);

        // Compute (Q + Q')x
        // Term 1: Qx - grad[i] += sum_j Q[i,j] * x[j]
        // Term 2: Q'x - grad[i] += sum_j Q[j,i] * x[j]
        for (i, grad_i) in grad.iter_mut().enumerate() {
            // Term 1: row i of Q times x
            if i < m {
                let cols = self.q_matrix[i].len().min(n);
                for (j, &xj) in x.iter().enumerate().take(cols) {
                    *grad_i += self.q_matrix[i][j] * xj;
                }
            }
            // Term 2: column i of Q times x (i.e., row i of Q')
            for (j, &xj) in x.iter().enumerate().take(m) {
                if i < self.q_matrix[j].len() {
                    *grad_i += self.q_matrix[j][i] * xj;
                }
            }
        }
        grad
    }

    /// Project x onto the constraint using gradient descent
    ///
    /// Uses iterative projection for quadratic constraints.
    /// For convex constraints (Q positive semi-definite), this converges.
    pub fn project(&self, x: &[f32], max_iters: usize, step_size: f32) -> Vec<f32> {
        if self.check(x) {
            return x.to_vec();
        }

        let mut current = x.to_vec();
        let n = current.len();

        for _ in 0..max_iters {
            let val = self.evaluate(&current);
            let grad = self.gradient(&current);

            // Compute step direction based on constraint type
            let step = match &self.constraint_type {
                QuadraticConstraintType::LessEq => {
                    if val <= self.rhs {
                        break;
                    }
                    // Move in negative gradient direction
                    val - self.rhs
                }
                QuadraticConstraintType::GreaterEq => {
                    if val >= self.rhs {
                        break;
                    }
                    // Move in positive gradient direction
                    self.rhs - val
                }
                QuadraticConstraintType::Equality { tolerance } => {
                    if (val - self.rhs).abs() <= *tolerance {
                        break;
                    }
                    val - self.rhs
                }
            };

            // Normalize gradient
            let grad_norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
            if grad_norm < f32::EPSILON {
                break;
            }

            // Update
            let factor = step_size * step / grad_norm;
            for i in 0..n {
                current[i] -= factor * grad[i];
            }

            // Check convergence
            if step.abs() < 1e-6 {
                break;
            }
        }

        current
    }

    /// Get the weight
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Get the Q matrix
    pub fn q_matrix(&self) -> &[Vec<f32>] {
        &self.q_matrix
    }

    /// Get the linear coefficients
    pub fn linear(&self) -> &[f32] {
        &self.linear
    }

    /// Get the right-hand side
    pub fn rhs(&self) -> f32 {
        self.rhs
    }
}

/// Set of quadratic constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadraticConstraintSet {
    constraints: Vec<QuadraticConstraint>,
}

impl QuadraticConstraintSet {
    /// Create a new constraint set
    pub fn new(constraints: Vec<QuadraticConstraint>) -> Self {
        Self { constraints }
    }

    /// Check if all constraints are satisfied
    pub fn check_all(&self, x: &[f32]) -> bool {
        self.constraints.iter().all(|c| c.check(x))
    }

    /// Compute total weighted violation
    pub fn total_violation(&self, x: &[f32]) -> f32 {
        self.constraints
            .iter()
            .map(|c| c.violation(x) * c.weight())
            .sum()
    }

    /// Project onto feasible region using alternating projections
    pub fn project(&self, x: &[f32], max_outer_iters: usize, max_inner_iters: usize) -> Vec<f32> {
        let mut current = x.to_vec();

        for _ in 0..max_outer_iters {
            let prev = current.clone();
            for c in &self.constraints {
                current = c.project(&current, max_inner_iters, 0.1);
            }
            // Check convergence
            let diff: f32 = current
                .iter()
                .zip(prev.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            if diff < 1e-6 {
                break;
            }
        }

        current
    }

    /// Get the number of constraints
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Add a constraint
    pub fn add(&mut self, constraint: QuadraticConstraint) {
        self.constraints.push(constraint);
    }

    /// Get constraints
    pub fn constraints(&self) -> &[QuadraticConstraint] {
        &self.constraints
    }
}

impl Default for QuadraticConstraintSet {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}
