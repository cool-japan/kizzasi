//! Model Predictive Control (MPC) with Receding Horizon Constraints
//!
//! This module implements Model Predictive Control for constrained optimization
//! over finite prediction horizons with rolling/receding updates.
//!
//! # Key Concepts
//!
//! - **Prediction Horizon**: Length of future trajectory to optimize
//! - **Control Horizon**: Length of control inputs to compute
//! - **Receding Horizon**: Optimization repeats at each time step
//! - **Terminal Constraints**: End-state requirements
//! - **Warm Starting**: Initialize from previous solution
//!
//! # Applications
//!
//! - Time-series forecasting with physics constraints
//! - Trajectory optimization for robotics
//! - Process control with safety guarantees
//! - Resource scheduling with temporal constraints

use crate::constraint::ViolationComputable;
use crate::error::{LogicError, LogicResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::VecDeque;

/// Maximum number of cyclic-projection sweeps used to enforce control constraints.
const CONTROL_PROJECTION_SWEEPS: usize = 200;

/// Finite-difference step used for the numerical constraint-penalty gradient.
const GRADIENT_EPSILON: f32 = 1e-5;

/// MPC Configuration
///
/// `prediction_horizon` (N) is the number of steps the plant is rolled out and
/// costed; `control_horizon` (M) is the number of control vectors that are
/// actually optimised. `M <= N` is required. Steps `M..N` reuse the last
/// optimised control (a "hold-last" move-blocking scheme).
#[derive(Debug, Clone)]
pub struct MPCConfig {
    /// Prediction horizon (time steps to predict)
    pub prediction_horizon: usize,
    /// Control horizon (time steps to optimize); must be `>= 1` and `<= prediction_horizon`
    pub control_horizon: usize,
    /// State dimension
    pub state_dim: usize,
    /// Control dimension
    pub control_dim: usize,
    /// Maximum iterations per solve
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f32,
    /// Use warm starting from previous solution
    pub warm_start: bool,
    /// Terminal cost weight
    pub terminal_weight: f32,
}

impl Default for MPCConfig {
    fn default() -> Self {
        Self {
            prediction_horizon: 10,
            control_horizon: 10,
            state_dim: 1,
            control_dim: 1,
            max_iterations: 100,
            tolerance: 1e-4,
            warm_start: true,
            terminal_weight: 1.0,
        }
    }
}

impl MPCConfig {
    /// Validate the horizon and dimension settings.
    ///
    /// Returns [`LogicError::InvalidInput`] when the configuration cannot
    /// describe a well-posed MPC problem: a zero control/prediction horizon, a
    /// control horizon longer than the prediction horizon, or a zero control
    /// dimension.
    pub fn validate(&self) -> LogicResult<()> {
        if self.prediction_horizon == 0 {
            return Err(LogicError::InvalidInput(
                "prediction_horizon must be >= 1".to_string(),
            ));
        }
        if self.control_horizon == 0 {
            return Err(LogicError::InvalidInput(
                "control_horizon must be >= 1".to_string(),
            ));
        }
        if self.control_horizon > self.prediction_horizon {
            return Err(LogicError::InvalidInput(format!(
                "control_horizon ({}) must not exceed prediction_horizon ({})",
                self.control_horizon, self.prediction_horizon
            )));
        }
        if self.control_dim == 0 {
            return Err(LogicError::InvalidInput(
                "control_dim must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Cost function for MPC
pub trait MPCCost: Send + Sync {
    /// Stage cost: L(x_t, u_t)
    fn stage_cost(&self, state: &Array1<f32>, control: &Array1<f32>, time_step: usize) -> f32;

    /// Terminal cost: Φ(x_N)
    fn terminal_cost(&self, state: &Array1<f32>) -> f32;

    /// Gradient of stage cost w.r.t. state
    fn stage_cost_grad_state(
        &self,
        state: &Array1<f32>,
        control: &Array1<f32>,
        time_step: usize,
    ) -> Array1<f32>;

    /// Gradient of stage cost w.r.t. control
    fn stage_cost_grad_control(
        &self,
        state: &Array1<f32>,
        control: &Array1<f32>,
        time_step: usize,
    ) -> Array1<f32>;
}

/// Dynamics model for MPC
pub trait DynamicsModel: Send + Sync {
    /// State transition: x_{t+1} = f(x_t, u_t)
    fn step(&self, state: &Array1<f32>, control: &Array1<f32>) -> Array1<f32>;

    /// Jacobian w.r.t. state: ∂f/∂x
    fn jacobian_state(&self, state: &Array1<f32>, control: &Array1<f32>) -> Array2<f32>;

    /// Jacobian w.r.t. control: ∂f/∂u
    fn jacobian_control(&self, state: &Array1<f32>, control: &Array1<f32>) -> Array2<f32>;
}

/// Quadratic cost function: ||x - x_ref||^2_Q + ||u - u_ref||^2_R
pub struct QuadraticCost {
    /// State reference trajectory (must be non-empty)
    pub x_ref: Vec<Array1<f32>>,
    /// Control reference
    pub u_ref: Array1<f32>,
    /// State cost matrix (diagonal)
    pub q_weights: Array1<f32>,
    /// Control cost matrix (diagonal)
    pub r_weights: Array1<f32>,
    /// Terminal cost matrix (diagonal)
    pub q_terminal: Array1<f32>,
}

impl QuadraticCost {
    /// Create a new quadratic cost
    pub fn new(
        x_ref: Vec<Array1<f32>>,
        u_ref: Array1<f32>,
        q_weights: Array1<f32>,
        r_weights: Array1<f32>,
        q_terminal: Array1<f32>,
    ) -> Self {
        Self {
            x_ref,
            u_ref,
            q_weights,
            r_weights,
            q_terminal,
        }
    }
}

impl MPCCost for QuadraticCost {
    fn stage_cost(&self, state: &Array1<f32>, control: &Array1<f32>, time_step: usize) -> f32 {
        let x_ref = if time_step < self.x_ref.len() {
            &self.x_ref[time_step]
        } else {
            self.x_ref.last().expect("x_ref must be non-empty")
        };

        let state_error = state - x_ref;
        let control_error = control - &self.u_ref;

        let state_cost: f32 = state_error
            .iter()
            .zip(self.q_weights.iter())
            .map(|(e, q)| e * e * q)
            .sum();

        let control_cost: f32 = control_error
            .iter()
            .zip(self.r_weights.iter())
            .map(|(e, r)| e * e * r)
            .sum();

        state_cost + control_cost
    }

    fn terminal_cost(&self, state: &Array1<f32>) -> f32 {
        let x_ref = self
            .x_ref
            .last()
            .expect("invariant: x_ref non-empty (set during construction)");
        let error = state - x_ref;

        error
            .iter()
            .zip(self.q_terminal.iter())
            .map(|(e, q)| e * e * q)
            .sum()
    }

    fn stage_cost_grad_state(
        &self,
        state: &Array1<f32>,
        _control: &Array1<f32>,
        time_step: usize,
    ) -> Array1<f32> {
        let x_ref = if time_step < self.x_ref.len() {
            &self.x_ref[time_step]
        } else {
            self.x_ref.last().expect("x_ref must be non-empty")
        };

        let error = state - x_ref;
        &error * &(&self.q_weights * 2.0)
    }

    fn stage_cost_grad_control(
        &self,
        _state: &Array1<f32>,
        control: &Array1<f32>,
        _time_step: usize,
    ) -> Array1<f32> {
        let error = control - &self.u_ref;
        &error * &(&self.r_weights * 2.0)
    }
}

/// Linear dynamics: x_{t+1} = A*x_t + B*u_t
pub struct LinearDynamics {
    /// State transition matrix A
    pub a_matrix: Array2<f32>,
    /// Control input matrix B
    pub b_matrix: Array2<f32>,
}

impl LinearDynamics {
    /// Create new linear dynamics
    pub fn new(a_matrix: Array2<f32>, b_matrix: Array2<f32>) -> Self {
        Self { a_matrix, b_matrix }
    }
}

impl DynamicsModel for LinearDynamics {
    fn step(&self, state: &Array1<f32>, control: &Array1<f32>) -> Array1<f32> {
        let ax = self.a_matrix.dot(state);
        let bu = self.b_matrix.dot(control);
        &ax + &bu
    }

    fn jacobian_state(&self, _state: &Array1<f32>, _control: &Array1<f32>) -> Array2<f32> {
        self.a_matrix.clone()
    }

    fn jacobian_control(&self, _state: &Array1<f32>, _control: &Array1<f32>) -> Array2<f32> {
        self.b_matrix.clone()
    }
}

/// MPC Controller
pub struct MPCController<D: DynamicsModel, C: MPCCost> {
    /// Configuration
    config: MPCConfig,
    /// Dynamics model
    dynamics: D,
    /// Cost function
    cost: C,
    /// State constraints
    state_constraints: Vec<Box<dyn ViolationComputable + Send + Sync>>,
    /// Control constraints
    control_constraints: Vec<Box<dyn ViolationComputable + Send + Sync>>,
    /// Previous control sequence (for warm starting)
    previous_controls: Option<VecDeque<Array1<f32>>>,
}

impl<D: DynamicsModel, C: MPCCost> MPCController<D, C> {
    /// Create a new MPC controller
    pub fn new(config: MPCConfig, dynamics: D, cost: C) -> Self {
        Self {
            config,
            dynamics,
            cost,
            state_constraints: Vec::new(),
            control_constraints: Vec::new(),
            previous_controls: None,
        }
    }

    /// Add a state constraint
    pub fn add_state_constraint(&mut self, constraint: Box<dyn ViolationComputable + Send + Sync>) {
        self.state_constraints.push(constraint);
    }

    /// Add a control constraint
    pub fn add_control_constraint(
        &mut self,
        constraint: Box<dyn ViolationComputable + Send + Sync>,
    ) {
        self.control_constraints.push(constraint);
    }

    /// Solve MPC problem for current state
    ///
    /// The plant is rolled out for `config.prediction_horizon` (N) steps while
    /// only `config.control_horizon` (M) control vectors are optimised; the
    /// last optimised control is held constant for the remaining `N - M` steps
    /// (hold-last move blocking). The returned
    /// [`MPCSolution::controls`] therefore has `M` entries and
    /// [`MPCSolution::predicted_states`] has `N + 1` entries.
    ///
    /// Every candidate control is projected onto the registered control
    /// constraints; if that projection cannot be carried out (no projection
    /// operator, or an empty constraint intersection) the solve fails with
    /// [`LogicError::ProjectionFailed`] instead of returning a control that
    /// violates the configured limits.
    pub fn solve(&mut self, current_state: &Array1<f32>) -> LogicResult<MPCSolution> {
        self.config.validate()?;

        let control_horizon = self.config.control_horizon;
        let prediction_horizon = self.config.prediction_horizon;
        let control_dim = self.config.control_dim;

        // Initialize the decision variables (one control vector per control step)
        let mut controls = match (self.config.warm_start, self.previous_controls.as_ref()) {
            (true, Some(prev_controls)) => {
                // Warm start from the previous solution (shift by one step) and
                // resize to the horizon/dimension currently configured.
                let mut shifted: Vec<Array1<f32>> = prev_controls
                    .iter()
                    .skip(1)
                    .map(|u| {
                        if u.len() == control_dim {
                            u.clone()
                        } else {
                            Array1::zeros(control_dim)
                        }
                    })
                    .collect();
                shifted.resize(control_horizon, Array1::zeros(control_dim));
                shifted
            }
            _ => vec![Array1::zeros(control_dim); control_horizon],
        };

        // Every control must start inside the feasible set, otherwise the very
        // first reported solution could violate the configured limits.
        for control in controls.iter_mut() {
            *control = self.project_control(control)?;
        }

        // Gradient descent optimization
        let step_size = 0.01;
        let mut best_cost = f32::INFINITY;

        for iteration in 0..self.config.max_iterations {
            // Forward simulate the full prediction horizon
            let applied = Self::expand_controls(&controls, prediction_horizon);
            let states = self.simulate_trajectory(current_state, &applied);

            // Compute total cost
            let cost = self.compute_total_cost(&states, &applied);

            if cost < best_cost {
                best_cost = cost;
            }

            // Check convergence
            if iteration > 0 && (best_cost - cost).abs() < self.config.tolerance {
                break;
            }

            // Compute gradient and update controls
            for (t, control) in controls.iter_mut().enumerate() {
                let grad = self.compute_control_gradient(&states, &applied, t, control_horizon)?;

                // Gradient descent step with projection
                let new_control = &*control - &(&grad * step_size);
                *control = self.project_control(&new_control)?;
            }
        }

        // Store for warm starting
        if self.config.warm_start {
            self.previous_controls = Some(controls.iter().cloned().collect());
        }

        // Simulate final trajectory over the full prediction horizon
        let applied = Self::expand_controls(&controls, prediction_horizon);
        let final_states = self.simulate_trajectory(current_state, &applied);
        let final_cost = self.compute_total_cost(&final_states, &applied);
        let constraint_violation = self.compute_raw_violation(&final_states, &applied);

        Ok(MPCSolution {
            controls,
            predicted_states: final_states,
            total_cost: final_cost,
            horizon: prediction_horizon,
            control_horizon,
            constraint_violation,
        })
    }

    /// Expand the `M` optimised controls into the `N` controls that are applied
    /// over the prediction horizon, holding the last optimised control constant
    /// for steps `M..N`.
    fn expand_controls(controls: &[Array1<f32>], prediction_horizon: usize) -> Vec<Array1<f32>> {
        let mut applied = Vec::with_capacity(prediction_horizon);
        for step in 0..prediction_horizon {
            let index = step.min(controls.len().saturating_sub(1));
            match controls.get(index) {
                Some(control) => applied.push(control.clone()),
                None => break,
            }
        }
        applied
    }

    /// Simulate trajectory forward
    fn simulate_trajectory(
        &self,
        initial_state: &Array1<f32>,
        controls: &[Array1<f32>],
    ) -> Vec<Array1<f32>> {
        let mut states = vec![initial_state.clone()];

        for control in controls.iter() {
            let next_state = self.dynamics.step(
                states
                    .last()
                    .expect("invariant: states initialized with initial_state"),
                control,
            );
            states.push(next_state);
        }

        states
    }

    /// Compute total cost over trajectory
    fn compute_total_cost(&self, states: &[Array1<f32>], controls: &[Array1<f32>]) -> f32 {
        let mut cost = 0.0;

        // Stage costs
        for (t, control) in controls.iter().enumerate() {
            let Some(state) = states.get(t) else { break };
            cost += self.cost.stage_cost(state, control, t);

            // Add constraint violations
            cost += self.constraint_violation_cost(state, control);
        }

        // Terminal cost
        cost += self.cost.terminal_cost(
            states
                .last()
                .expect("invariant: states initialized with initial_state"),
        ) * self.config.terminal_weight;

        cost
    }

    /// Compute constraint violation cost
    fn constraint_violation_cost(&self, state: &Array1<f32>, control: &Array1<f32>) -> f32 {
        let mut violation = 0.0;

        let state_slice: Vec<f32> = state.iter().copied().collect();
        for constraint in &self.state_constraints {
            violation += constraint.violation(&state_slice) * 100.0; // High penalty
        }

        let control_slice: Vec<f32> = control.iter().copied().collect();
        for constraint in &self.control_constraints {
            violation += constraint.violation(&control_slice) * 100.0;
        }

        violation
    }

    /// Compute raw (unscaled) constraint violation sum over a trajectory.
    ///
    /// Unlike `constraint_violation_cost`, this does **not** multiply by the
    /// penalty factor (×100).  It is used at solve-time to populate
    /// `MPCSolution::constraint_violation` so that `is_feasible()` can make a
    /// meaningful comparison against a numerical tolerance.
    fn compute_raw_violation(&self, states: &[Array1<f32>], controls: &[Array1<f32>]) -> f32 {
        let mut v = 0.0_f32;
        for (t, control) in controls.iter().enumerate() {
            let Some(state) = states.get(t) else { break };
            let s: Vec<f32> = state.iter().copied().collect();
            for c in &self.state_constraints {
                v += c.violation(&s);
            }
            let u: Vec<f32> = control.iter().copied().collect();
            for c in &self.control_constraints {
                v += c.violation(&u);
            }
        }
        v
    }

    /// Compute the gradient of the cost w.r.t. the `t`-th *decision* control.
    ///
    /// `controls` is the expanded (applied) sequence of length
    /// `prediction_horizon`. Decision control `t` drives prediction step `t`,
    /// and the last decision control is additionally held for every step in
    /// `control_horizon..prediction_horizon`, so its gradient accumulates the
    /// contribution of all held steps.
    fn compute_control_gradient(
        &self,
        states: &[Array1<f32>],
        controls: &[Array1<f32>],
        t: usize,
        control_horizon: usize,
    ) -> LogicResult<Array1<f32>> {
        let control_dim = self.config.control_dim;
        let mut gradient = Array1::<f32>::zeros(control_dim);

        let last_decision = control_horizon.saturating_sub(1);
        let end = if t == last_decision {
            controls.len()
        } else {
            (t + 1).min(controls.len())
        };

        for step in t..end {
            let (Some(state), Some(control)) = (states.get(step), controls.get(step)) else {
                break;
            };

            // Direct gradient from the cost function
            let direct = self.cost.stage_cost_grad_control(state, control, step);
            if direct.len() != control_dim {
                return Err(LogicError::DimensionMismatch {
                    expected: control_dim,
                    got: direct.len(),
                });
            }
            gradient = &gradient + &direct;

            // Numerical gradient of the constraint penalty. The unperturbed
            // penalty is invariant across the control dimensions, so it is
            // evaluated once per step rather than once per dimension.
            let base = self.constraint_violation_cost(state, control);
            let mut perturbed = control.clone();
            for i in 0..control_dim {
                let Some(original) = perturbed.get(i).copied() else {
                    break;
                };
                if let Some(slot) = perturbed.get_mut(i) {
                    *slot = original + GRADIENT_EPSILON;
                }
                let plus = self.constraint_violation_cost(state, &perturbed);
                if let Some(slot) = perturbed.get_mut(i) {
                    *slot = original;
                }
                if let Some(slot) = gradient.get_mut(i) {
                    *slot += (plus - base) / GRADIENT_EPSILON;
                }
            }
        }

        Ok(gradient)
    }

    /// Project a control vector onto the registered control constraint set.
    ///
    /// Uses cyclic projection (POCS): each violated constraint is replaced by
    /// its own projection operator, sweeping until every constraint is
    /// satisfied within `config.tolerance`. For a single constraint this is the
    /// exact Euclidean projection.
    ///
    /// # Errors
    ///
    /// * [`LogicError::ProjectionFailed`] when a violated constraint offers no
    ///   projection operator ([`ViolationComputable::project_onto`] returned
    ///   `None`), or when the sweeps do not reach a feasible point (an empty or
    ///   numerically empty constraint intersection).
    /// * [`LogicError::DimensionMismatch`] when a projection returns a vector
    ///   of a different length.
    fn project_control(&self, control: &Array1<f32>) -> LogicResult<Array1<f32>> {
        if self.control_constraints.is_empty() {
            return Ok(control.clone());
        }

        let tolerance = self.config.tolerance.max(f32::EPSILON);
        let mut current: Vec<f32> = control.iter().copied().collect();

        for _ in 0..CONTROL_PROJECTION_SWEEPS {
            let mut max_move = 0.0_f32;

            for constraint in &self.control_constraints {
                if constraint.violation(&current) <= tolerance {
                    continue;
                }

                let projected = constraint.project_onto(&current).ok_or_else(|| {
                    LogicError::ProjectionFailed(
                        "control constraint has no projection operator \
                         (ViolationComputable::project_onto returned None)"
                            .to_string(),
                    )
                })?;

                if projected.len() != current.len() {
                    return Err(LogicError::DimensionMismatch {
                        expected: current.len(),
                        got: projected.len(),
                    });
                }

                max_move = max_move.max(
                    projected
                        .iter()
                        .zip(current.iter())
                        .map(|(p, c)| (p - c).abs())
                        .fold(0.0_f32, f32::max),
                );
                current = projected;
            }

            if self.total_control_violation(&current) <= tolerance {
                return Ok(Array1::from_vec(current));
            }

            if max_move <= tolerance {
                // The sweep stalled: no constraint could move the point any
                // further, yet the point is still infeasible.
                break;
            }
        }

        let residual = self.total_control_violation(&current);
        if residual <= tolerance {
            Ok(Array1::from_vec(current))
        } else {
            Err(LogicError::ProjectionFailed(format!(
                "control projection did not converge: residual violation {residual} \
                 exceeds tolerance {tolerance}"
            )))
        }
    }

    /// Sum of all control-constraint violations at `control`.
    fn total_control_violation(&self, control: &[f32]) -> f32 {
        self.control_constraints
            .iter()
            .map(|constraint| constraint.violation(control))
            .sum()
    }

    /// Reset warm start cache
    pub fn reset(&mut self) {
        self.previous_controls = None;
    }
}

/// MPC Solution
///
/// `controls` holds the `control_horizon` optimised control vectors; the last
/// one is held constant for the remaining prediction steps, so
/// `predicted_states` has `horizon + 1` entries.
///
/// Control constraints registered on the controller are hard-enforced by
/// projection, so `controls` satisfies them to within `MPCConfig::tolerance`
/// (the solve fails rather than returning a control that does not). State
/// constraints are only penalised in the cost — their residual is reported
/// through `constraint_violation`, not eliminated.
#[derive(Debug, Clone)]
pub struct MPCSolution {
    /// Optimised control sequence (`control_horizon` entries)
    pub controls: Vec<Array1<f32>>,
    /// Predicted state trajectory (`horizon + 1` entries)
    pub predicted_states: Vec<Array1<f32>>,
    /// Total cost
    pub total_cost: f32,
    /// Prediction horizon length (number of simulated steps)
    pub horizon: usize,
    /// Control horizon length (number of optimised control vectors)
    pub control_horizon: usize,
    /// Raw sum of all constraint violations (without penalty scaling)
    pub constraint_violation: f32,
}

impl MPCSolution {
    /// Get first control (to be applied), or `None` for an empty sequence
    pub fn first_control(&self) -> Option<&Array1<f32>> {
        self.controls.first()
    }

    /// Get predicted state at time step
    pub fn predicted_state(&self, time_step: usize) -> Option<&Array1<f32>> {
        self.predicted_states.get(time_step)
    }

    /// Check if all constraints are satisfied within numerical tolerance.
    ///
    /// `constraint_violation` is a **sum** over every predicted step, while the
    /// threshold here is absolute. Control constraints are projected to (near)
    /// exact feasibility so they contribute ~0; the residual this reports is
    /// therefore dominated by the state constraints, which are only penalised
    /// in the cost and never projected.
    pub fn is_feasible(&self) -> bool {
        self.total_cost.is_finite() && self.constraint_violation <= 1e-4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::ConstraintBuilder;

    #[test]
    fn test_mpc_config() {
        let config = MPCConfig::default();
        assert_eq!(config.prediction_horizon, 10);
        assert_eq!(config.control_horizon, 10);
        assert!(config.warm_start);
    }

    #[test]
    fn test_quadratic_cost() {
        let x_ref = vec![Array1::from_vec(vec![1.0])];
        let u_ref = Array1::from_vec(vec![0.0]);
        let q = Array1::from_vec(vec![1.0]);
        let r = Array1::from_vec(vec![0.1]);
        let q_term = Array1::from_vec(vec![10.0]);

        let cost = QuadraticCost::new(x_ref, u_ref, q, r, q_term);

        let state = Array1::from_vec(vec![2.0]);
        let control = Array1::from_vec(vec![1.0]);

        let stage = cost.stage_cost(&state, &control, 0);
        assert!(stage > 0.0); // (2-1)^2 * 1 + 1^2 * 0.1 = 1.1

        let terminal = cost.terminal_cost(&state);
        assert!(terminal > 0.0); // (2-1)^2 * 10 = 10
    }

    #[test]
    fn test_linear_dynamics() {
        let a = Array2::from_shape_vec((1, 1), vec![0.9]).unwrap();
        let b = Array2::from_shape_vec((1, 1), vec![0.1]).unwrap();

        let dynamics = LinearDynamics::new(a, b);

        let state = Array1::from_vec(vec![1.0]);
        let control = Array1::from_vec(vec![2.0]);

        let next_state = dynamics.step(&state, &control);
        assert!((next_state[0] - 1.1).abs() < 1e-5); // 0.9*1 + 0.1*2 = 1.1
    }

    #[test]
    fn test_mpc_controller_creation() {
        let config = MPCConfig {
            prediction_horizon: 5,
            control_horizon: 5,
            state_dim: 1,
            control_dim: 1,
            ..Default::default()
        };

        let a = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let b = Array2::from_shape_vec((1, 1), vec![0.1]).unwrap();
        let dynamics = LinearDynamics::new(a, b);

        let x_ref = vec![Array1::from_vec(vec![0.0]); 5];
        let u_ref = Array1::from_vec(vec![0.0]);
        let q = Array1::from_vec(vec![1.0]);
        let r = Array1::from_vec(vec![0.1]);
        let q_term = Array1::from_vec(vec![10.0]);
        let cost = QuadraticCost::new(x_ref, u_ref, q, r, q_term);

        let mut mpc = MPCController::new(config, dynamics, cost);

        let initial_state = Array1::from_vec(vec![1.0]);
        let solution = mpc.solve(&initial_state).unwrap();

        assert_eq!(solution.controls.len(), 5);
        assert_eq!(solution.predicted_states.len(), 6); // N+1 states
        assert!(solution.total_cost < f32::INFINITY);
    }

    #[test]
    fn test_mpc_warm_start() {
        let config = MPCConfig {
            prediction_horizon: 3,
            control_horizon: 3,
            state_dim: 1,
            control_dim: 1,
            warm_start: true,
            ..Default::default()
        };

        let a = Array2::from_shape_vec((1, 1), vec![0.95]).unwrap();
        let b = Array2::from_shape_vec((1, 1), vec![0.05]).unwrap();
        let dynamics = LinearDynamics::new(a, b);

        let x_ref = vec![Array1::from_vec(vec![0.0]); 3];
        let u_ref = Array1::from_vec(vec![0.0]);
        let q = Array1::from_vec(vec![1.0]);
        let r = Array1::from_vec(vec![0.01]);
        let q_term = Array1::from_vec(vec![5.0]);
        let cost = QuadraticCost::new(x_ref, u_ref, q, r, q_term);

        let mut mpc = MPCController::new(config, dynamics, cost);

        // First solve
        let state1 = Array1::from_vec(vec![1.0]);
        let _sol1 = mpc.solve(&state1).unwrap();

        assert!(mpc.previous_controls.is_some());

        // Second solve (should use warm start)
        let state2 = Array1::from_vec(vec![0.9]);
        let _sol2 = mpc.solve(&state2).unwrap();
    }

    #[test]
    fn test_mpc_solution_methods() {
        let controls = vec![
            Array1::from_vec(vec![1.0]),
            Array1::from_vec(vec![0.5]),
            Array1::from_vec(vec![0.2]),
        ];

        let states = vec![
            Array1::from_vec(vec![0.0]),
            Array1::from_vec(vec![0.1]),
            Array1::from_vec(vec![0.15]),
            Array1::from_vec(vec![0.17]),
        ];

        let solution = MPCSolution {
            controls,
            predicted_states: states,
            total_cost: 1.5,
            horizon: 3,
            control_horizon: 3,
            constraint_violation: 0.0,
        };

        assert_eq!(
            solution.first_control().expect("first control present")[0],
            1.0
        );
        assert_eq!(solution.predicted_state(0).unwrap()[0], 0.0);
        assert_eq!(solution.predicted_state(2).unwrap()[0], 0.15);
        assert!(solution.is_feasible());
    }

    #[test]
    fn test_mpc_is_feasible_discriminates() {
        // Both solutions have finite total_cost; the old tautological check
        // would have returned true for both.  The corrected check must
        // distinguish them via constraint_violation.
        let dummy_controls = vec![Array1::zeros(1)];
        let dummy_states = vec![Array1::zeros(1), Array1::zeros(1)];

        let feasible = MPCSolution {
            controls: dummy_controls.clone(),
            predicted_states: dummy_states.clone(),
            total_cost: 2.0,
            horizon: 1,
            control_horizon: 1,
            constraint_violation: 0.0,
        };

        let infeasible = MPCSolution {
            controls: dummy_controls,
            predicted_states: dummy_states,
            total_cost: 502.0, // finite but violating (penalty already baked in)
            horizon: 1,
            control_horizon: 1,
            constraint_violation: 5.0,
        };

        assert!(feasible.is_feasible());
        assert!(!infeasible.is_feasible());
    }

    #[test]
    fn test_mpc_is_feasible_rejects_nonfinite() {
        let dummy_controls = vec![Array1::zeros(1)];
        let dummy_states = vec![Array1::zeros(1), Array1::zeros(1)];

        let inf_solution = MPCSolution {
            controls: dummy_controls.clone(),
            predicted_states: dummy_states.clone(),
            total_cost: f32::INFINITY,
            horizon: 1,
            control_horizon: 1,
            constraint_violation: 0.0,
        };
        assert!(!inf_solution.is_feasible());

        let nan_solution = MPCSolution {
            controls: dummy_controls,
            predicted_states: dummy_states,
            total_cost: f32::NAN,
            horizon: 1,
            control_horizon: 1,
            constraint_violation: 0.0,
        };
        assert!(!nan_solution.is_feasible());
    }

    /// Build a scalar integrator (x_{t+1} = x_t + u_t) driven toward
    /// `reference` with control weight `control_weight`. A large weight makes
    /// the very first gradient step ask for a large control, which is what
    /// exercises the projection.
    fn integrator_controller(
        config: MPCConfig,
        reference: f32,
        control_weight: f32,
    ) -> MPCController<LinearDynamics, QuadraticCost> {
        let a = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("1x1 A");
        let b = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("1x1 B");
        let dynamics = LinearDynamics::new(a, b);

        let cost = QuadraticCost::new(
            vec![Array1::from_vec(vec![reference])],
            Array1::from_vec(vec![reference]),
            Array1::from_vec(vec![1.0_f32]),
            Array1::from_vec(vec![control_weight]),
            Array1::from_vec(vec![1.0_f32]),
        );

        MPCController::new(config, dynamics, cost)
    }

    /// Regression (finding 126/299): a control bound of |u| <= 1 must be
    /// honoured even though the requested control is 5.0.
    ///
    /// The old `project_control` clamped to a hardcoded `[-10, 10]`, so a
    /// control of 5.0 passed through untouched.
    #[test]
    fn test_mpc_honours_tight_control_bound() {
        let config = MPCConfig {
            prediction_horizon: 4,
            control_horizon: 4,
            state_dim: 1,
            control_dim: 1,
            warm_start: false,
            ..Default::default()
        };

        // Reference far away => the unconstrained optimum wants |u| >> 1.
        let mut mpc = integrator_controller(config, 50.0, 100.0);

        let bound = ConstraintBuilder::new()
            .name("u_bound")
            .in_range(-1.0, 1.0)
            .build()
            .expect("bound builds");
        mpc.add_control_constraint(Box::new(bound));

        let solution = mpc
            .solve(&Array1::from_vec(vec![0.0_f32]))
            .expect("constrained solve");

        for (t, control) in solution.controls.iter().enumerate() {
            assert!(
                control[0] <= 1.0 + 1e-4 && control[0] >= -1.0 - 1e-4,
                "control at step {t} must stay inside [-1, 1], got {}",
                control[0]
            );
        }
        assert!(
            solution.constraint_violation <= 1e-3,
            "control constraints must be satisfied, residual {}",
            solution.constraint_violation
        );
    }

    /// Regression (finding 126/299): a control bound wider than the old
    /// hardcoded `[-10, 10]` box must not be truncated to it.
    #[test]
    fn test_mpc_wide_control_bound_not_truncated() {
        let config = MPCConfig {
            prediction_horizon: 3,
            control_horizon: 3,
            state_dim: 1,
            control_dim: 1,
            warm_start: false,
            max_iterations: 400,
            ..Default::default()
        };

        // Reference at 100 => the optimiser drives u toward 100, well past 10.
        let mut mpc = integrator_controller(config, 200.0, 100.0);

        let bound = ConstraintBuilder::new()
            .name("u_wide")
            .in_range(-100.0, 100.0)
            .build()
            .expect("bound builds");
        mpc.add_control_constraint(Box::new(bound));

        let solution = mpc
            .solve(&Array1::from_vec(vec![0.0_f32]))
            .expect("constrained solve");

        let u0 = solution.controls[0][0];
        assert!(
            u0 > 10.0,
            "a [-100, 100] bound must not be truncated to the old [-10, 10] box, got {u0}"
        );
        assert!(u0 <= 100.0 + 1e-3, "u0 must respect the real bound: {u0}");
    }

    /// A control constraint whose type offers no projection operator must make
    /// the solve fail loudly rather than silently returning an unprojected
    /// control.
    #[test]
    fn test_mpc_reports_missing_projection_operator() {
        struct NoProjection;
        impl ViolationComputable for NoProjection {
            fn violation(&self, x: &[f32]) -> f32 {
                // Always violated by a fixed amount.
                let _ = x;
                1.0
            }
            fn check(&self, _x: &[f32]) -> bool {
                false
            }
        }

        let config = MPCConfig {
            prediction_horizon: 2,
            control_horizon: 2,
            state_dim: 1,
            control_dim: 1,
            warm_start: false,
            ..Default::default()
        };
        let mut mpc = integrator_controller(config, 1.0, 1.0);
        mpc.add_control_constraint(Box::new(NoProjection));

        let result = mpc.solve(&Array1::from_vec(vec![0.0_f32]));
        assert!(
            matches!(result, Err(LogicError::ProjectionFailed(_))),
            "unprojectable control constraint must surface ProjectionFailed"
        );
    }

    /// Regression (finding 257): `prediction_horizon` must drive the rollout
    /// length while `control_horizon` drives the number of optimised controls.
    #[test]
    fn test_mpc_prediction_horizon_drives_rollout() {
        let config = MPCConfig {
            prediction_horizon: 12,
            control_horizon: 3,
            state_dim: 1,
            control_dim: 1,
            warm_start: false,
            ..Default::default()
        };
        let mut mpc = integrator_controller(config, 0.0, 1.0);

        let solution = mpc
            .solve(&Array1::from_vec(vec![1.0_f32]))
            .expect("solve succeeds");

        assert_eq!(
            solution.controls.len(),
            3,
            "controls must have control_horizon entries"
        );
        assert_eq!(
            solution.predicted_states.len(),
            13,
            "trajectory must have prediction_horizon + 1 states"
        );
        assert_eq!(
            solution.horizon, 12,
            "reported horizon is the prediction horizon"
        );
        assert_eq!(solution.control_horizon, 3);
    }

    /// Regression (finding 257): warm starting after a horizon change must
    /// resize the cached control sequence instead of reusing a stale length.
    #[test]
    fn test_mpc_warm_start_resizes_to_control_horizon() {
        let config = MPCConfig {
            prediction_horizon: 6,
            control_horizon: 4,
            state_dim: 1,
            control_dim: 1,
            warm_start: true,
            ..Default::default()
        };
        let mut mpc = integrator_controller(config, 0.0, 1.0);

        let first = mpc
            .solve(&Array1::from_vec(vec![1.0_f32]))
            .expect("first solve");
        assert_eq!(first.controls.len(), 4);

        // The warm-start cache is shifted by one; the next solve must still
        // return exactly control_horizon controls.
        let second = mpc
            .solve(&Array1::from_vec(vec![0.8_f32]))
            .expect("second solve");
        assert_eq!(
            second.controls.len(),
            4,
            "warm start must be resized back to control_horizon"
        );
    }

    /// An invalid horizon configuration must be rejected instead of silently
    /// producing a shorter prediction.
    #[test]
    fn test_mpc_rejects_control_horizon_longer_than_prediction() {
        let config = MPCConfig {
            prediction_horizon: 2,
            control_horizon: 5,
            state_dim: 1,
            control_dim: 1,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let mut mpc = integrator_controller(config, 0.0, 1.0);
        let result = mpc.solve(&Array1::from_vec(vec![1.0_f32]));
        assert!(
            matches!(result, Err(LogicError::InvalidInput(_))),
            "control_horizon > prediction_horizon must be rejected"
        );
    }
}
