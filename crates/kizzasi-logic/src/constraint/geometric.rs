use serde::{Deserialize, Serialize};

// ============================================================================
// Set Membership Constraints
// ============================================================================

/// Geometric set types for membership constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeometricSet {
    /// Axis-aligned box: l <= x <= u
    Box { lower: Vec<f32>, upper: Vec<f32> },
    /// Euclidean ball: ||x - center||₂ <= radius
    Ball { center: Vec<f32>, radius: f32 },
    /// Ellipsoid: (x-c)ᵀ P (x-c) <= 1
    Ellipsoid {
        center: Vec<f32>,
        /// Inverse covariance matrix (flattened)
        shape_inv: Vec<f32>,
    },
    /// Polytope: Ax <= b
    Polytope {
        /// Constraint matrix A (row-major)
        a_matrix: Vec<f32>,
        /// Right-hand side b
        b_vector: Vec<f32>,
        /// Number of rows in A
        num_constraints: usize,
        /// Number of columns in A (dimension)
        dimension: usize,
    },
    /// L-infinity ball: ||x - center||_∞ <= radius
    LInfBall { center: Vec<f32>, radius: f32 },
    /// Simplex: x_i >= 0, Σx_i <= 1
    Simplex { dimension: usize },
}

impl GeometricSet {
    /// Create a box constraint
    pub fn box_constraint(lower: Vec<f32>, upper: Vec<f32>) -> Self {
        assert_eq!(
            lower.len(),
            upper.len(),
            "Lower and upper bounds must have same dimension"
        );
        Self::Box { lower, upper }
    }

    /// Create a ball constraint
    pub fn ball(center: Vec<f32>, radius: f32) -> Self {
        assert!(radius > 0.0, "Radius must be positive");
        Self::Ball { center, radius }
    }

    /// Create an ellipsoid constraint
    pub fn ellipsoid(center: Vec<f32>, shape_inv: Vec<f32>) -> Self {
        let dim = center.len();
        assert_eq!(shape_inv.len(), dim * dim, "Shape matrix must be dim × dim");
        Self::Ellipsoid { center, shape_inv }
    }

    /// Create a polytope constraint Ax <= b
    pub fn polytope(
        a_matrix: Vec<f32>,
        b_vector: Vec<f32>,
        num_constraints: usize,
        dimension: usize,
    ) -> Self {
        assert_eq!(
            a_matrix.len(),
            num_constraints * dimension,
            "A matrix size mismatch"
        );
        assert_eq!(b_vector.len(), num_constraints, "b vector size mismatch");
        Self::Polytope {
            a_matrix,
            b_vector,
            num_constraints,
            dimension,
        }
    }

    /// Create an L-infinity ball
    pub fn l_inf_ball(center: Vec<f32>, radius: f32) -> Self {
        assert!(radius > 0.0, "Radius must be positive");
        Self::LInfBall { center, radius }
    }

    /// Create a simplex constraint
    pub fn simplex(dimension: usize) -> Self {
        Self::Simplex { dimension }
    }

    /// Check if a point is in the set
    pub fn contains(&self, x: &[f32]) -> bool {
        match self {
            Self::Box { lower, upper } => x
                .iter()
                .zip(lower.iter())
                .zip(upper.iter())
                .all(|((&xi, &li), &ui)| xi >= li && xi <= ui),
            Self::Ball { center, radius } => {
                let dist_sq: f32 = x
                    .iter()
                    .zip(center.iter())
                    .map(|(&xi, &ci)| (xi - ci).powi(2))
                    .sum();
                dist_sq <= radius * radius
            }
            Self::Ellipsoid { center, shape_inv } => {
                let dim = center.len();
                let diff: Vec<f32> = x
                    .iter()
                    .zip(center.iter())
                    .map(|(&xi, &ci)| xi - ci)
                    .collect();

                // Compute (x-c)ᵀ P (x-c)
                let mut quad_form = 0.0;
                for i in 0..dim {
                    for j in 0..dim {
                        quad_form += diff[i] * shape_inv[i * dim + j] * diff[j];
                    }
                }
                quad_form <= 1.0
            }
            Self::Polytope {
                a_matrix,
                b_vector,
                num_constraints,
                dimension,
            } => {
                for i in 0..*num_constraints {
                    let mut ax = 0.0;
                    for j in 0..*dimension {
                        ax += a_matrix[i * dimension + j] * x[j];
                    }
                    if ax > b_vector[i] {
                        return false;
                    }
                }
                true
            }
            Self::LInfBall { center, radius } => x
                .iter()
                .zip(center.iter())
                .all(|(&xi, &ci)| (xi - ci).abs() <= *radius),
            Self::Simplex { dimension } => {
                if x.len() != *dimension {
                    return false;
                }
                let sum: f32 = x.iter().sum();
                x.iter().all(|&xi| xi >= 0.0) && sum <= 1.0
            }
        }
    }

    /// Compute distance to the set (0 if inside)
    pub fn distance(&self, x: &[f32]) -> f32 {
        match self {
            Self::Box { lower, upper } => x
                .iter()
                .zip(lower.iter())
                .zip(upper.iter())
                .map(|((&xi, &li), &ui)| {
                    if xi < li {
                        li - xi
                    } else if xi > ui {
                        xi - ui
                    } else {
                        0.0
                    }
                })
                .map(|d| d * d)
                .sum::<f32>()
                .sqrt(),
            Self::Ball { center, radius } => {
                let dist_sq: f32 = x
                    .iter()
                    .zip(center.iter())
                    .map(|(&xi, &ci)| (xi - ci).powi(2))
                    .sum();
                let dist = dist_sq.sqrt();
                (dist - radius).max(0.0)
            }
            Self::Ellipsoid { .. } => {
                // For ellipsoid, use simple check-based distance
                if self.contains(x) {
                    0.0
                } else {
                    // Approximation: would need proper optimization
                    1.0
                }
            }
            Self::Polytope { .. } => {
                // For polytope, use simple check-based distance
                if self.contains(x) {
                    0.0
                } else {
                    1.0
                }
            }
            Self::LInfBall { center, radius } => {
                let max_diff = x
                    .iter()
                    .zip(center.iter())
                    .map(|(&xi, &ci)| (xi - ci).abs())
                    .fold(0.0f32, |a, b| a.max(b));
                (max_diff - radius).max(0.0)
            }
            Self::Simplex { dimension } => {
                if x.len() != *dimension {
                    return f32::MAX;
                }
                let neg_sum: f32 = x.iter().filter(|&&xi| xi < 0.0).map(|&xi| -xi).sum();
                let sum: f32 = x.iter().sum();
                let excess = (sum - 1.0).max(0.0);
                (neg_sum.powi(2) + excess.powi(2)).sqrt()
            }
        }
    }

    /// Project a point onto the set
    pub fn project(&self, x: &[f32]) -> Vec<f32> {
        match self {
            Self::Box { lower, upper } => x
                .iter()
                .zip(lower.iter())
                .zip(upper.iter())
                .map(|((&xi, &li), &ui)| xi.clamp(li, ui))
                .collect(),
            Self::Ball { center, radius } => {
                let diff: Vec<f32> = x
                    .iter()
                    .zip(center.iter())
                    .map(|(&xi, &ci)| xi - ci)
                    .collect();
                let dist_sq: f32 = diff.iter().map(|&d| d * d).sum();
                let dist = dist_sq.sqrt();

                if dist <= *radius {
                    x.to_vec()
                } else {
                    center
                        .iter()
                        .zip(diff.iter())
                        .map(|(&ci, &di)| ci + di * radius / dist)
                        .collect()
                }
            }
            Self::Ellipsoid { .. } | Self::Polytope { .. } => {
                // For ellipsoid and polytope, proper projection requires optimization/QP solver
                // Simple fallback: return input (would need iterative projection for exact solution)
                x.to_vec()
            }
            Self::LInfBall { center, radius } => x
                .iter()
                .zip(center.iter())
                .map(|(&xi, &ci)| {
                    let diff = xi - ci;
                    ci + diff.clamp(-*radius, *radius)
                })
                .collect(),
            Self::Simplex { dimension } => {
                if x.len() != *dimension {
                    return x.to_vec();
                }

                // Project onto simplex using efficient algorithm
                let mut sorted: Vec<f32> = x.to_vec();
                sorted.sort_by(|a, b| b.total_cmp(a));

                let mut theta = 0.0;
                let mut t_sum = 0.0;

                for (i, &val) in sorted.iter().enumerate() {
                    t_sum += val;
                    let candidate = (t_sum - 1.0) / (i + 1) as f32;
                    if i + 1 == sorted.len() || sorted[i + 1] < val - candidate {
                        theta = candidate;
                        break;
                    }
                }

                x.iter().map(|&xi| (xi - theta).max(0.0)).collect()
            }
        }
    }
}

/// Set membership constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetMembershipConstraint {
    name: String,
    set: GeometricSet,
    weight: f32,
}

impl SetMembershipConstraint {
    /// Create a new set membership constraint
    pub fn new(name: impl Into<String>, set: GeometricSet) -> Self {
        Self {
            name: name.into(),
            set,
            weight: 1.0,
        }
    }

    /// Set the constraint weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Check if point is in the set
    pub fn check(&self, x: &[f32]) -> bool {
        self.set.contains(x)
    }

    /// Compute violation (distance to set)
    pub fn violation(&self, x: &[f32]) -> f32 {
        self.set.distance(x)
    }

    /// Project point onto the set
    pub fn project(&self, x: &[f32]) -> Vec<f32> {
        self.set.project(x)
    }

    /// Get constraint name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get weight
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Get the geometric set
    pub fn set(&self) -> &GeometricSet {
        &self.set
    }
}
