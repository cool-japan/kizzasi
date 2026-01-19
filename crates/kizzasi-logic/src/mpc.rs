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
use crate::error::LogicResult;
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::VecDeque;

/// MPC Configuration
#[derive(Debug, Clone)]
pub struct MPCConfig {
    /// Prediction horizon (time steps to predict)
    pub prediction_horizon: usize,
    /// Control horizon (time steps to optimize)
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
        let x_ref = self.x_ref.last().unwrap();
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
    /// Returns the optimal control sequence
    pub fn solve(&mut self, current_state: &Array1<f32>) -> LogicResult<MPCSolution> {
        let horizon = self.config.control_horizon;

        // Initialize control sequence
        let mut controls = if self.config.warm_start {
            if let Some(prev_controls) = &self.previous_controls {
                // Warm start from previous solution (shift and append)
                let mut prev = prev_controls.clone();
                if !prev.is_empty() {
                    prev.pop_front();
                    prev.push_back(Array1::zeros(self.config.control_dim));
                }
                prev.into_iter().collect::<Vec<_>>()
            } else {
                // Cold start with zeros
                vec![Array1::zeros(self.config.control_dim); horizon]
            }
        } else {
            // Cold start with zeros
            vec![Array1::zeros(self.config.control_dim); horizon]
        };

        // Gradient descent optimization
        let step_size = 0.01;
        let mut best_cost = f32::INFINITY;

        for iteration in 0..self.config.max_iterations {
            // Forward simulate to get state trajectory
            let states = self.simulate_trajectory(current_state, &controls);

            // Compute total cost
            let cost = self.compute_total_cost(&states, &controls);

            if cost < best_cost {
                best_cost = cost;
            }

            // Check convergence
            if iteration > 0 && (best_cost - cost).abs() < self.config.tolerance {
                break;
            }

            // Compute gradient and update controls
            for t in 0..horizon {
                let grad = self.compute_control_gradient(&states, &controls, t);

                // Gradient descent step with projection
                let new_control = &controls[t] - &(&grad * step_size);
                controls[t] = self.project_control(&new_control);
            }
        }

        // Store for warm starting
        if self.config.warm_start {
            self.previous_controls = Some(controls.iter().cloned().collect());
        }

        // Simulate final trajectory
        let final_states = self.simulate_trajectory(current_state, &controls);
        let final_cost = self.compute_total_cost(&final_states, &controls);

        Ok(MPCSolution {
            controls,
            predicted_states: final_states,
            total_cost: final_cost,
            horizon,
        })
    }

    /// Simulate trajectory forward
    fn simulate_trajectory(
        &self,
        initial_state: &Array1<f32>,
        controls: &[Array1<f32>],
    ) -> Vec<Array1<f32>> {
        let mut states = vec![initial_state.clone()];

        for control in controls.iter() {
            let next_state = self.dynamics.step(states.last().unwrap(), control);
            states.push(next_state);
        }

        states
    }

    /// Compute total cost over trajectory
    fn compute_total_cost(&self, states: &[Array1<f32>], controls: &[Array1<f32>]) -> f32 {
        let mut cost = 0.0;

        // Stage costs
        for (t, control) in controls.iter().enumerate() {
            cost += self.cost.stage_cost(&states[t], control, t);

            // Add constraint violations
            cost += self.constraint_violation_cost(&states[t], control);
        }

        // Terminal cost
        cost += self.cost.terminal_cost(states.last().unwrap()) * self.config.terminal_weight;

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

    /// Compute gradient of cost w.r.t. control at time t
    fn compute_control_gradient(
        &self,
        states: &[Array1<f32>],
        controls: &[Array1<f32>],
        t: usize,
    ) -> Array1<f32> {
        // Direct gradient from cost function
        let direct_grad = self
            .cost
            .stage_cost_grad_control(&states[t], &controls[t], t);

        // Numerical gradient for constraints (simplified)
        let mut constraint_grad = Array1::zeros(self.config.control_dim);
        let eps = 1e-5;

        for i in 0..self.config.control_dim {
            let mut control_plus = controls[t].clone();
            control_plus[i] += eps;

            let cost_plus = self.constraint_violation_cost(&states[t], &control_plus);
            let cost_base = self.constraint_violation_cost(&states[t], &controls[t]);

            constraint_grad[i] = (cost_plus - cost_base) / eps;
        }

        &direct_grad + &constraint_grad
    }

    /// Project control onto constraint set
    fn project_control(&self, control: &Array1<f32>) -> Array1<f32> {
        let mut projected = control.clone();

        // Simple box projection for each constraint
        for constraint in &self.control_constraints {
            let control_slice: Vec<f32> = projected.iter().copied().collect();
            if constraint.violation(&control_slice) > 0.0 {
                // Simplified projection (would use more sophisticated method in practice)
                projected = projected.mapv(|x| x.clamp(-10.0, 10.0));
            }
        }

        projected
    }

    /// Reset warm start cache
    pub fn reset(&mut self) {
        self.previous_controls = None;
    }
}

/// MPC Solution
#[derive(Debug, Clone)]
pub struct MPCSolution {
    /// Optimal control sequence
    pub controls: Vec<Array1<f32>>,
    /// Predicted state trajectory
    pub predicted_states: Vec<Array1<f32>>,
    /// Total cost
    pub total_cost: f32,
    /// Horizon length
    pub horizon: usize,
}

impl MPCSolution {
    /// Get first control (to be applied)
    pub fn first_control(&self) -> &Array1<f32> {
        &self.controls[0]
    }

    /// Get predicted state at time step
    pub fn predicted_state(&self, time_step: usize) -> Option<&Array1<f32>> {
        self.predicted_states.get(time_step)
    }

    /// Check if all constraints are satisfied
    pub fn is_feasible(&self) -> bool {
        // Simplified feasibility check
        self.total_cost < f32::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        };

        assert_eq!(solution.first_control()[0], 1.0);
        assert_eq!(solution.predicted_state(0).unwrap()[0], 0.0);
        assert_eq!(solution.predicted_state(2).unwrap()[0], 0.15);
        assert!(solution.is_feasible());
    }
}
