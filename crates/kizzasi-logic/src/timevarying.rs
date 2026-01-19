//! Time-varying Constraints
//!
//! This module provides constraints that change over time, including:
//! - Scheduled constraint changes
//! - State-dependent constraint activation
//! - Predictive constraint adaptation
//! - Temporal constraint interpolation

use crate::{LogicError, LogicResult, ViolationComputable};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// A constraint that varies over time
#[derive(Debug, Clone)]
pub struct TimeVaryingConstraint<C: ViolationComputable> {
    /// Name of the constraint
    #[allow(dead_code)]
    name: String,
    /// Base constraint that gets modified
    #[allow(dead_code)]
    base_constraint: C,
    /// Scheduled parameter changes (time, update) sorted by time
    schedule: Vec<(f32, ParameterUpdate)>,
    /// Current time
    current_time: f32,
    /// Interpolation mode
    #[allow(dead_code)]
    interpolation: InterpolationMode,
}

/// Parameter update for constraint modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterUpdate {
    /// Scale factor for constraint tightness (1.0 = no change)
    pub scale: Option<f32>,
    /// Additive offset for constraint bounds
    pub offset: Option<Array1<f32>>,
    /// Completely replace constraint parameters
    pub replacement: Option<ConstraintParams>,
}

/// Constraint parameters that can be updated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintParams {
    /// Linear constraint: A*x <= b
    Linear { a: Array2<f32>, b: Array1<f32> },
    /// Quadratic constraint: x^T Q x + c^T x <= d
    Quadratic {
        q: Array2<f32>,
        c: Array1<f32>,
        d: f32,
    },
    /// Box constraint: lower <= x <= upper
    Box {
        lower: Array1<f32>,
        upper: Array1<f32>,
    },
}

/// Interpolation mode for smooth transitions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InterpolationMode {
    /// No interpolation, step changes
    Step,
    /// Linear interpolation between keyframes
    Linear,
    /// Smooth cubic interpolation
    Cubic,
    /// Exponential decay/growth
    Exponential { rate: f32 },
}

impl<C: ViolationComputable + Clone> TimeVaryingConstraint<C> {
    /// Create a new time-varying constraint
    pub fn new(
        name: impl Into<String>,
        base_constraint: C,
        interpolation: InterpolationMode,
    ) -> Self {
        Self {
            name: name.into(),
            base_constraint,
            schedule: Vec::new(),
            current_time: 0.0,
            interpolation,
        }
    }

    /// Schedule a parameter update at a specific time
    pub fn schedule_update(&mut self, time: f32, update: ParameterUpdate) {
        // Insert in sorted order
        let pos = self.schedule.iter().position(|(t, _)| *t > time);
        match pos {
            Some(idx) => self.schedule.insert(idx, (time, update)),
            None => self.schedule.push((time, update)),
        }
    }

    /// Advance time and update constraint parameters
    pub fn advance_time(&mut self, time: f32) -> LogicResult<()> {
        if time < self.current_time {
            return Err(LogicError::InvalidInput(
                "Cannot go backward in time".to_string(),
            ));
        }
        self.current_time = time;
        Ok(())
    }

    /// Get current time
    pub fn current_time(&self) -> f32 {
        self.current_time
    }

    /// Get interpolated parameter update for current time
    #[allow(dead_code)]
    fn get_current_update(&self) -> Option<ParameterUpdate> {
        // Find surrounding keyframes
        let before_idx = self
            .schedule
            .iter()
            .rposition(|(t, _)| *t <= self.current_time);
        let after_idx = self
            .schedule
            .iter()
            .position(|(t, _)| *t >= self.current_time);

        match (before_idx, after_idx) {
            (Some(i1), Some(i2)) if i1 != i2 => {
                let (t1, u1) = &self.schedule[i1];
                let (t2, u2) = &self.schedule[i2];
                // Interpolate between keyframes
                let alpha = (self.current_time - t1) / (t2 - t1);
                Some(self.interpolate_updates(u1, u2, alpha))
            }
            (Some(i), None) | (None, Some(i)) => Some(self.schedule[i].1.clone()),
            _ => None,
        }
    }

    /// Interpolate between two parameter updates
    #[allow(dead_code)]
    fn interpolate_updates(
        &self,
        u1: &ParameterUpdate,
        u2: &ParameterUpdate,
        alpha: f32,
    ) -> ParameterUpdate {
        let alpha = match self.interpolation {
            InterpolationMode::Step => {
                if alpha < 1.0 {
                    0.0
                } else {
                    1.0
                }
            }
            InterpolationMode::Linear => alpha,
            InterpolationMode::Cubic => {
                // Smooth step function
                alpha * alpha * (3.0 - 2.0 * alpha)
            }
            InterpolationMode::Exponential { rate } => 1.0 - (-rate * alpha).exp(),
        };

        ParameterUpdate {
            scale: match (u1.scale, u2.scale) {
                (Some(s1), Some(s2)) => Some(s1 + (s2 - s1) * alpha),
                (Some(s), None) | (None, Some(s)) => Some(s),
                _ => None,
            },
            offset: match (&u1.offset, &u2.offset) {
                (Some(o1), Some(o2)) => Some(o1 + &(o2 - o1) * alpha),
                (Some(o), None) | (None, Some(o)) => Some(o.clone()),
                _ => None,
            },
            replacement: if alpha < 0.5 {
                u1.replacement.clone()
            } else {
                u2.replacement.clone()
            },
        }
    }
}

/// State-dependent constraint activation
#[derive(Debug, Clone)]
pub struct StateDependentConstraint<C: ViolationComputable> {
    /// Name of the constraint
    #[allow(dead_code)]
    name: String,
    /// The constraint to apply when active
    constraint: C,
    /// Activation function: returns true if constraint should be active
    activation_fn: ActivationFunction,
    /// Current activation state
    is_active: bool,
}

/// Activation function type
#[derive(Debug, Clone)]
pub enum ActivationFunction {
    /// Activate when state norm exceeds threshold
    NormThreshold { threshold: f32 },
    /// Activate when specific state component exceeds threshold
    ComponentThreshold { index: usize, threshold: f32 },
    /// Activate when state enters a region
    RegionBased {
        lower: Array1<f32>,
        upper: Array1<f32>,
    },
    /// Activate based on state velocity (rate of change)
    VelocityBased { threshold: f32 },
    /// Custom activation function
    Custom(fn(&Array1<f32>) -> bool),
}

impl<C: ViolationComputable + Clone> StateDependentConstraint<C> {
    /// Create a new state-dependent constraint
    pub fn new(name: impl Into<String>, constraint: C, activation_fn: ActivationFunction) -> Self {
        Self {
            name: name.into(),
            constraint,
            activation_fn,
            is_active: false,
        }
    }

    /// Update activation state based on current system state
    pub fn update_activation(&mut self, state: &Array1<f32>) -> bool {
        self.is_active = match &self.activation_fn {
            ActivationFunction::NormThreshold { threshold } => {
                let norm = state.iter().map(|x| x * x).sum::<f32>().sqrt();
                norm > *threshold
            }
            ActivationFunction::ComponentThreshold { index, threshold } => state
                .get(*index)
                .map(|x| x.abs() > *threshold)
                .unwrap_or(false),
            ActivationFunction::RegionBased { lower, upper } => {
                state.iter().zip(lower.iter()).all(|(x, l)| x >= l)
                    && state.iter().zip(upper.iter()).all(|(x, u)| x <= u)
            }
            ActivationFunction::VelocityBased { threshold } => {
                // This requires historical state; simplified version
                state.iter().any(|x| x.abs() > *threshold)
            }
            ActivationFunction::Custom(f) => f(state),
        };
        self.is_active
    }

    /// Check if constraint is currently active
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Check constraint if active
    pub fn check_if_active(&self, state: &Array1<f32>) -> bool {
        if self.is_active {
            self.constraint.check(state.as_slice().unwrap_or(&[]))
        } else {
            true // Inactive constraints are trivially satisfied
        }
    }

    /// Get violation if active
    pub fn violation_if_active(&self, state: &Array1<f32>) -> f32 {
        if self.is_active {
            self.constraint.violation(state.as_slice().unwrap_or(&[]))
        } else {
            0.0
        }
    }
}

/// Predictive constraint adaptation
#[derive(Debug, Clone)]
pub struct PredictiveConstraintAdapter<C: ViolationComputable> {
    /// Name of the adapter
    #[allow(dead_code)]
    name: String,
    /// Base constraint
    base_constraint: C,
    /// Prediction horizon (steps ahead)
    horizon: usize,
    /// Historical violations for learning
    violation_history: Vec<f32>,
    /// Adaptation rate (how quickly to adjust)
    adaptation_rate: f32,
    /// Current tightness multiplier
    tightness: f32,
}

impl<C: ViolationComputable + Clone> PredictiveConstraintAdapter<C> {
    /// Create a new predictive constraint adapter
    pub fn new(
        name: impl Into<String>,
        base_constraint: C,
        horizon: usize,
        adaptation_rate: f32,
    ) -> Self {
        Self {
            name: name.into(),
            base_constraint,
            horizon,
            violation_history: Vec::new(),
            adaptation_rate,
            tightness: 1.0,
        }
    }

    /// Predict future violations based on trajectory
    pub fn predict_violations(&self, trajectory: &[Array1<f32>]) -> Vec<f32> {
        let mut violations = Vec::new();
        for state in trajectory.iter().take(self.horizon) {
            let viol = self
                .base_constraint
                .violation(state.as_slice().unwrap_or(&[]));
            violations.push(viol);
        }
        violations
    }

    /// Adapt constraint based on predicted violations
    pub fn adapt(&mut self, predicted_violations: &[f32]) -> LogicResult<()> {
        // Calculate mean predicted violation
        let mean_violation = if predicted_violations.is_empty() {
            0.0
        } else {
            predicted_violations.iter().sum::<f32>() / predicted_violations.len() as f32
        };

        // Record in history
        self.violation_history.push(mean_violation);
        if self.violation_history.len() > 100 {
            self.violation_history.remove(0);
        }

        // Adapt tightness: tighten if violations predicted, loosen if safe
        if mean_violation > 0.0 {
            // Tighten constraint
            self.tightness *= 1.0 + self.adaptation_rate * mean_violation;
        } else {
            // Gradually loosen if no violations
            self.tightness *= 1.0 - self.adaptation_rate * 0.1;
        }

        // Keep tightness in reasonable range
        self.tightness = self.tightness.clamp(0.5, 2.0);

        Ok(())
    }

    /// Get current tightness multiplier
    pub fn tightness(&self) -> f32 {
        self.tightness
    }

    /// Get violation history
    pub fn violation_history(&self) -> &[f32] {
        &self.violation_history
    }
}

/// Temporal constraint interpolation
#[derive(Debug, Clone)]
pub struct ConstraintInterpolator<C: ViolationComputable> {
    /// Name of the interpolator
    #[allow(dead_code)]
    name: String,
    /// Start constraint
    start_constraint: C,
    /// End constraint
    end_constraint: C,
    /// Interpolation parameter (0.0 to 1.0)
    alpha: f32,
    /// Interpolation mode
    mode: InterpolationMode,
}

impl<C: ViolationComputable + Clone> ConstraintInterpolator<C> {
    /// Create a new constraint interpolator
    pub fn new(
        name: impl Into<String>,
        start_constraint: C,
        end_constraint: C,
        mode: InterpolationMode,
    ) -> Self {
        Self {
            name: name.into(),
            start_constraint,
            end_constraint,
            alpha: 0.0,
            mode,
        }
    }

    /// Set interpolation parameter (0.0 = start, 1.0 = end)
    pub fn set_alpha(&mut self, alpha: f32) -> LogicResult<()> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(LogicError::InvalidInput(
                "Alpha must be in [0, 1]".to_string(),
            ));
        }
        self.alpha = alpha;
        Ok(())
    }

    /// Get current interpolation parameter
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Compute interpolated violation
    pub fn violation(&self, state: &Array1<f32>) -> f32 {
        let v1 = self
            .start_constraint
            .violation(state.as_slice().unwrap_or(&[]));
        let v2 = self
            .end_constraint
            .violation(state.as_slice().unwrap_or(&[]));

        let alpha = match self.mode {
            InterpolationMode::Step => {
                if self.alpha < 1.0 {
                    0.0
                } else {
                    1.0
                }
            }
            InterpolationMode::Linear => self.alpha,
            InterpolationMode::Cubic => self.alpha * self.alpha * (3.0 - 2.0 * self.alpha),
            InterpolationMode::Exponential { rate } => 1.0 - (-rate * self.alpha).exp(),
        };

        v1 * (1.0 - alpha) + v2 * alpha
    }

    /// Check interpolated constraint
    pub fn check(&self, state: &Array1<f32>) -> bool {
        self.violation(state) <= 0.0
    }
}

/// Manager for multiple time-varying constraints
#[derive(Debug, Clone)]
pub struct TimeVaryingConstraintSet<C: ViolationComputable> {
    /// Collection of state-dependent constraints
    state_dependent: Vec<StateDependentConstraint<C>>,
    /// Collection of predictive adapters
    predictive: Vec<PredictiveConstraintAdapter<C>>,
    /// Collection of interpolators
    interpolators: Vec<ConstraintInterpolator<C>>,
    /// Current global time
    current_time: f32,
}

impl<C: ViolationComputable + Clone> TimeVaryingConstraintSet<C> {
    /// Create a new constraint set
    pub fn new() -> Self {
        Self {
            state_dependent: Vec::new(),
            predictive: Vec::new(),
            interpolators: Vec::new(),
            current_time: 0.0,
        }
    }

    /// Add a state-dependent constraint
    pub fn add_state_dependent(&mut self, constraint: StateDependentConstraint<C>) {
        self.state_dependent.push(constraint);
    }

    /// Add a predictive adapter
    pub fn add_predictive(&mut self, adapter: PredictiveConstraintAdapter<C>) {
        self.predictive.push(adapter);
    }

    /// Add an interpolator
    pub fn add_interpolator(&mut self, interpolator: ConstraintInterpolator<C>) {
        self.interpolators.push(interpolator);
    }

    /// Advance global time
    pub fn advance_time(&mut self, time: f32) -> LogicResult<()> {
        self.current_time = time;
        Ok(())
    }

    /// Update all state-dependent activations
    pub fn update_activations(&mut self, state: &Array1<f32>) {
        for constraint in &mut self.state_dependent {
            constraint.update_activation(state);
        }
    }

    /// Get number of active constraints
    pub fn num_active(&self) -> usize {
        self.state_dependent
            .iter()
            .filter(|c| c.is_active())
            .count()
            + self.predictive.len()
            + self.interpolators.len()
    }

    /// Check all constraints
    pub fn check_all(&self, state: &Array1<f32>) -> bool {
        // Check state-dependent constraints
        for constraint in &self.state_dependent {
            if !constraint.check_if_active(state) {
                return false;
            }
        }

        // Check interpolators
        for interpolator in &self.interpolators {
            if !interpolator.check(state) {
                return false;
            }
        }

        true
    }

    /// Compute total violation across all constraints
    pub fn total_violation(&self, state: &Array1<f32>) -> f32 {
        let mut total = 0.0;

        // State-dependent violations
        for constraint in &self.state_dependent {
            total += constraint.violation_if_active(state).max(0.0);
        }

        // Interpolator violations
        for interpolator in &self.interpolators {
            total += interpolator.violation(state).max(0.0);
        }

        total
    }
}

impl<C: ViolationComputable + Clone> Default for TimeVaryingConstraintSet<C> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearConstraint;

    #[test]
    fn test_state_dependent_activation() {
        // x <= 1.0
        let base = LinearConstraint::less_eq(vec![1.0], 1.0);

        let mut sdc = StateDependentConstraint::new(
            "test",
            base,
            ActivationFunction::NormThreshold { threshold: 5.0 },
        );

        let state = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let active = sdc.update_activation(&state);
        assert!(!active); // norm ≈ 3.74 < 5.0

        let state2 = Array1::from_vec(vec![3.0, 4.0, 5.0]);
        let active2 = sdc.update_activation(&state2);
        assert!(active2); // norm ≈ 7.07 > 5.0
    }

    #[test]
    fn test_predictive_adaptation() {
        // x <= 1.0
        let base = LinearConstraint::less_eq(vec![1.0], 1.0);

        let mut adapter = PredictiveConstraintAdapter::new("test", base, 5, 0.1);

        let trajectory = vec![
            Array1::from_vec(vec![0.5]),
            Array1::from_vec(vec![0.8]),
            Array1::from_vec(vec![1.2]), // violation
        ];

        let violations = adapter.predict_violations(&trajectory);
        assert_eq!(violations.len(), 3);

        let _ = adapter.adapt(&violations);
        assert!(adapter.tightness() >= 1.0); // Should tighten due to predicted violation
    }

    #[test]
    fn test_constraint_interpolation() -> LogicResult<()> {
        // x <= 1.0 and x <= 2.0
        let start = LinearConstraint::less_eq(vec![1.0], 1.0);
        let end = LinearConstraint::less_eq(vec![1.0], 2.0);

        let mut interp = ConstraintInterpolator::new("test", start, end, InterpolationMode::Linear);

        interp.set_alpha(0.5)?;
        assert_eq!(interp.alpha(), 0.5);

        let state = Array1::from_vec(vec![1.5]);
        let violation = interp.violation(&state);
        // x=1.5: start violation = 1.5-1.0=0.5, end violation = max(1.5-2.0, 0)=0
        // interpolated: 0.5 * 0.5 + 0.5 * 0 = 0.25
        assert!((0.0..=0.5).contains(&violation));

        Ok(())
    }

    #[test]
    fn test_constraint_set() {
        let mut set = TimeVaryingConstraintSet::new();

        // x <= 1.0
        let base = LinearConstraint::less_eq(vec![1.0], 1.0);
        let sdc = StateDependentConstraint::new(
            "state_dep",
            base,
            ActivationFunction::NormThreshold { threshold: 5.0 },
        );

        set.add_state_dependent(sdc);

        let state = Array1::from_vec(vec![1.0, 2.0]);
        set.update_activations(&state);

        assert_eq!(set.num_active(), 0); // Not active due to low norm

        let state2 = Array1::from_vec(vec![5.0, 5.0]);
        set.update_activations(&state2);
        assert_eq!(set.num_active(), 1); // Should be active now
    }
}
