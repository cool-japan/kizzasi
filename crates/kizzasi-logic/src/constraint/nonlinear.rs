use serde::{Deserialize, Serialize};

// ============================================================================
// Nonlinear Constraints
// ============================================================================

/// Type alias for constraint function
type ConstraintFn = std::sync::Arc<dyn Fn(&[f32]) -> f32 + Send + Sync>;

/// Type alias for gradient function
type GradientFn = std::sync::Arc<dyn Fn(&[f32]) -> Vec<f32> + Send + Sync>;

/// Nonlinear constraint defined by an arbitrary function
///
/// Represents a constraint of the form: f(x) <= 0 or f(x) == 0
/// where f is a user-defined nonlinear function.
#[derive(Clone)]
pub struct NonlinearConstraint {
    /// Constraint name for debugging
    name: String,
    /// Constraint function: f(x) <= 0 for inequality, f(x) == 0 for equality
    constraint_fn: ConstraintFn,
    /// Optional gradient function: ∇f(x)
    gradient_fn: Option<GradientFn>,
    /// Constraint type
    constraint_type: NonlinearConstraintType,
    /// Weight for loss computation
    weight: f32,
}

impl std::fmt::Debug for NonlinearConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NonlinearConstraint")
            .field("name", &self.name)
            .field("has_gradient", &self.gradient_fn.is_some())
            .field("constraint_type", &self.constraint_type)
            .field("weight", &self.weight)
            .finish()
    }
}

/// Type of nonlinear constraint
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NonlinearConstraintType {
    /// Inequality: f(x) <= 0
    Inequality,
    /// Equality: |f(x)| <= tolerance
    Equality { tolerance: f32 },
}

impl NonlinearConstraint {
    /// Create a new nonlinear inequality constraint: f(x) <= 0
    pub fn inequality<F>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(&[f32]) -> f32 + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            constraint_fn: std::sync::Arc::new(f),
            gradient_fn: None,
            constraint_type: NonlinearConstraintType::Inequality,
            weight: 1.0,
        }
    }

    /// Create a new nonlinear equality constraint: |f(x)| <= tolerance
    pub fn equality<F>(name: impl Into<String>, f: F, tolerance: f32) -> Self
    where
        F: Fn(&[f32]) -> f32 + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            constraint_fn: std::sync::Arc::new(f),
            gradient_fn: None,
            constraint_type: NonlinearConstraintType::Equality { tolerance },
            weight: 1.0,
        }
    }

    /// Add gradient function for gradient-based projection
    pub fn with_gradient<G>(mut self, grad: G) -> Self
    where
        G: Fn(&[f32]) -> Vec<f32> + Send + Sync + 'static,
    {
        self.gradient_fn = Some(std::sync::Arc::new(grad));
        self
    }

    /// Set the constraint weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Evaluate the constraint function
    pub fn evaluate(&self, x: &[f32]) -> f32 {
        (self.constraint_fn)(x)
    }

    /// Evaluate the gradient (if available)
    pub fn gradient(&self, x: &[f32]) -> Option<Vec<f32>> {
        self.gradient_fn.as_ref().map(|g| g(x))
    }

    /// Check if constraint is satisfied
    pub fn check(&self, x: &[f32]) -> bool {
        let val = self.evaluate(x);
        match self.constraint_type {
            NonlinearConstraintType::Inequality => val <= 0.0,
            NonlinearConstraintType::Equality { tolerance } => val.abs() <= tolerance,
        }
    }

    /// Compute violation amount (0 if satisfied)
    pub fn violation(&self, x: &[f32]) -> f32 {
        let val = self.evaluate(x);
        match self.constraint_type {
            NonlinearConstraintType::Inequality => val.max(0.0),
            NonlinearConstraintType::Equality { tolerance } => (val.abs() - tolerance).max(0.0),
        }
    }

    /// Project onto constraint using gradient descent (requires gradient function)
    pub fn project(&self, x: &[f32], max_iter: usize, step_size: f32) -> Vec<f32> {
        if let Some(grad_fn) = &self.gradient_fn {
            let mut result = x.to_vec();

            for _ in 0..max_iter {
                if self.check(&result) {
                    break;
                }

                let grad = grad_fn(&result);
                let val = self.evaluate(&result);

                // Gradient descent step
                for (ri, &gi) in result.iter_mut().zip(grad.iter()) {
                    *ri -= step_size * val.signum() * gi;
                }
            }

            result
        } else {
            // Without gradient, cannot project - return input
            x.to_vec()
        }
    }

    /// Get constraint name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get weight
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Check if gradient is available
    pub fn has_gradient(&self) -> bool {
        self.gradient_fn.is_some()
    }
}
