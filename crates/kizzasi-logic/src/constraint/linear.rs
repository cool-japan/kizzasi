use serde::{Deserialize, Serialize};

// ============================================================================
// Linear Constraints
// ============================================================================

/// Type of linear constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LinearConstraintType {
    /// a·x <= b
    LessEq,
    /// a·x >= b
    GreaterEq,
    /// |a·x - b| <= tolerance
    Equality { tolerance: f32 },
}

/// Linear constraint: a·x (op) b
///
/// Represents a single linear constraint on a vector of values.
/// - Inequality: a·x <= b or a·x >= b
/// - Equality: |a·x - b| <= tolerance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearConstraint {
    /// Coefficient vector `a`
    coefficients: Vec<f32>,
    /// Right-hand side `b`
    rhs: f32,
    /// Constraint type
    constraint_type: LinearConstraintType,
    /// Weight for loss computation
    weight: f32,
}

impl LinearConstraint {
    /// Create a less-than-or-equal constraint: a·x <= b
    pub fn less_eq(coefficients: Vec<f32>, rhs: f32) -> Self {
        Self {
            coefficients,
            rhs,
            constraint_type: LinearConstraintType::LessEq,
            weight: 1.0,
        }
    }

    /// Create a greater-than-or-equal constraint: a·x >= b
    pub fn greater_eq(coefficients: Vec<f32>, rhs: f32) -> Self {
        Self {
            coefficients,
            rhs,
            constraint_type: LinearConstraintType::GreaterEq,
            weight: 1.0,
        }
    }

    /// Create an equality constraint: |a·x - b| <= tolerance
    pub fn equality(coefficients: Vec<f32>, rhs: f32, tolerance: f32) -> Self {
        Self {
            coefficients,
            rhs,
            constraint_type: LinearConstraintType::Equality { tolerance },
            weight: 1.0,
        }
    }

    /// Set the constraint weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Compute dot product a·x
    fn dot(&self, x: &[f32]) -> f32 {
        self.coefficients
            .iter()
            .zip(x.iter())
            .map(|(a, xi)| a * xi)
            .sum()
    }

    /// Compute squared norm of coefficients ||a||²
    fn norm_sq(&self) -> f32 {
        self.coefficients.iter().map(|a| a * a).sum()
    }

    /// Check if values satisfy the constraint
    pub fn check(&self, x: &[f32]) -> bool {
        let ax = self.dot(x);
        match &self.constraint_type {
            LinearConstraintType::LessEq => ax <= self.rhs,
            LinearConstraintType::GreaterEq => ax >= self.rhs,
            LinearConstraintType::Equality { tolerance } => (ax - self.rhs).abs() <= *tolerance,
        }
    }

    /// Compute violation amount (0 if satisfied)
    pub fn violation(&self, x: &[f32]) -> f32 {
        let ax = self.dot(x);
        match &self.constraint_type {
            LinearConstraintType::LessEq => (ax - self.rhs).max(0.0),
            LinearConstraintType::GreaterEq => (self.rhs - ax).max(0.0),
            LinearConstraintType::Equality { tolerance } => {
                let diff = (ax - self.rhs).abs();
                (diff - tolerance).max(0.0)
            }
        }
    }

    /// Project x onto the constraint (closest point that satisfies)
    ///
    /// For a·x <= b or a·x >= b, uses orthogonal projection onto hyperplane.
    pub fn project(&self, x: &[f32]) -> Vec<f32> {
        let ax = self.dot(x);
        let norm_sq = self.norm_sq();

        if norm_sq < f32::EPSILON {
            return x.to_vec();
        }

        let needs_projection = match &self.constraint_type {
            LinearConstraintType::LessEq => ax > self.rhs,
            LinearConstraintType::GreaterEq => ax < self.rhs,
            LinearConstraintType::Equality { tolerance } => (ax - self.rhs).abs() > *tolerance,
        };

        if !needs_projection {
            return x.to_vec();
        }

        // Orthogonal projection: x' = x - ((a·x - b) / ||a||²) * a
        let factor = (ax - self.rhs) / norm_sq;
        x.iter()
            .zip(self.coefficients.iter())
            .map(|(xi, ai)| xi - factor * ai)
            .collect()
    }

    /// Get the weight
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }

    /// Get the right-hand side
    pub fn rhs(&self) -> f32 {
        self.rhs
    }
}

/// A set of linear constraints
///
/// Useful for representing polyhedral constraints (intersection of half-spaces).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearConstraintSet {
    constraints: Vec<LinearConstraint>,
}

impl LinearConstraintSet {
    /// Create a new constraint set
    pub fn new(constraints: Vec<LinearConstraint>) -> Self {
        Self { constraints }
    }

    /// Check if all constraints are satisfied
    pub fn check_all(&self, x: &[f32]) -> bool {
        self.constraints.iter().all(|c| c.check(x))
    }

    /// Get individual check results for each constraint
    pub fn check_each(&self, x: &[f32]) -> Vec<bool> {
        self.constraints.iter().map(|c| c.check(x)).collect()
    }

    /// Compute total weighted violation
    pub fn total_violation(&self, x: &[f32]) -> f32 {
        self.constraints
            .iter()
            .map(|c| c.violation(x) * c.weight())
            .sum()
    }

    /// Project x onto feasible region using cyclic projection (Dykstra's)
    ///
    /// Iteratively projects onto each constraint. May not converge to
    /// true projection for non-convex feasible regions.
    pub fn project(&self, x: &[f32], max_iters: usize) -> Vec<f32> {
        let mut current = x.to_vec();

        for _ in 0..max_iters {
            let prev = current.clone();
            for c in &self.constraints {
                current = c.project(&current);
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

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Add a constraint to the set
    pub fn add(&mut self, constraint: LinearConstraint) {
        self.constraints.push(constraint);
    }

    /// Get constraints
    pub fn constraints(&self) -> &[LinearConstraint] {
        &self.constraints
    }
}

impl Default for LinearConstraintSet {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

/// Affine equality constraint: Ax = b
///
/// Represents a system of linear equality constraints where A is an m×n matrix
/// and b is an m-vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineEquality {
    /// Matrix A stored row-major (m rows, each with n elements)
    matrix: Vec<Vec<f32>>,
    /// Right-hand side vector b (m elements)
    rhs: Vec<f32>,
    /// Tolerance for equality check
    tolerance: f32,
}

impl AffineEquality {
    /// Create a new affine equality constraint Ax = b
    ///
    /// # Arguments
    /// * `matrix` - Matrix A as row-major vec of vecs (m rows × n cols)
    /// * `rhs` - Vector b with m elements
    /// * `tolerance` - Tolerance for equality check
    pub fn new(matrix: Vec<Vec<f32>>, rhs: Vec<f32>, tolerance: f32) -> Self {
        Self {
            matrix,
            rhs,
            tolerance,
        }
    }

    /// Compute Ax
    fn multiply(&self, x: &[f32]) -> Vec<f32> {
        self.matrix
            .iter()
            .map(|row| row.iter().zip(x.iter()).map(|(a, xi)| a * xi).sum())
            .collect()
    }

    /// Check if Ax ≈ b (within tolerance)
    pub fn check(&self, x: &[f32]) -> bool {
        let ax = self.multiply(x);
        ax.iter()
            .zip(self.rhs.iter())
            .all(|(axi, bi)| (axi - bi).abs() <= self.tolerance)
    }

    /// Compute residual ||Ax - b||
    pub fn residual(&self, x: &[f32]) -> f32 {
        let ax = self.multiply(x);
        ax.iter()
            .zip(self.rhs.iter())
            .map(|(axi, bi)| (axi - bi).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Compute element-wise violations (Ax - b)
    pub fn violations(&self, x: &[f32]) -> Vec<f32> {
        let ax = self.multiply(x);
        ax.iter()
            .zip(self.rhs.iter())
            .map(|(axi, bi)| (axi - bi).abs())
            .collect()
    }

    /// Get the matrix
    pub fn matrix(&self) -> &[Vec<f32>] {
        &self.matrix
    }

    /// Get the right-hand side
    pub fn rhs(&self) -> &[f32] {
        &self.rhs
    }

    /// Get the number of equations (rows)
    pub fn num_equations(&self) -> usize {
        self.matrix.len()
    }

    /// Get the number of variables (columns)
    pub fn num_variables(&self) -> Option<usize> {
        self.matrix.first().map(|row| row.len())
    }
}
