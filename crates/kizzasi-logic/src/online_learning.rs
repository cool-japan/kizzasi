//! Online Constraint Learning
//!
//! This module provides algorithms for learning constraints from streaming data:
//! - Incremental constraint refinement
//! - Anomaly-based constraint discovery
//! - Constraint parameter tuning from feedback
//! - Active learning for constraint boundaries

use crate::{LinearConstraint, LogicResult};
use scirs2_core::ndarray::Array1;
use std::collections::VecDeque;

/// Online learner for refining constraints from streaming data
#[derive(Debug, Clone)]
pub struct OnlineConstraintLearner {
    /// Current constraint estimate
    constraint: LinearConstraint,
    /// Learning rate for parameter updates
    #[allow(dead_code)]
    learning_rate: f32,
    /// Historical data buffer
    data_buffer: VecDeque<(Array1<f32>, bool)>, // (sample, is_feasible)
    /// Maximum buffer size
    #[allow(dead_code)]
    max_buffer_size: usize,
    /// Number of updates performed
    update_count: usize,
}

impl OnlineConstraintLearner {
    /// Create a new online constraint learner
    pub fn new(
        initial_constraint: LinearConstraint,
        learning_rate: f32,
        max_buffer_size: usize,
    ) -> Self {
        Self {
            constraint: initial_constraint,
            learning_rate,
            data_buffer: VecDeque::new(),
            max_buffer_size,
            update_count: 0,
        }
    }

    /// Add a new observation and update the constraint
    pub fn observe(&mut self, sample: Array1<f32>, is_feasible: bool) -> LogicResult<()> {
        // Add to buffer
        self.data_buffer.push_back((sample.clone(), is_feasible));
        if self.data_buffer.len() > self.max_buffer_size {
            self.data_buffer.pop_front();
        }

        // Update constraint using stochastic gradient descent
        self.refine_constraint(&sample, is_feasible)?;
        self.update_count += 1;

        Ok(())
    }

    /// Refine constraint based on a single observation
    fn refine_constraint(&mut self, sample: &Array1<f32>, is_feasible: bool) -> LogicResult<()> {
        let sample_slice = sample.as_slice().unwrap_or(&[]);
        let current_satisfied = self.constraint.check(sample_slice);

        // If prediction matches observation, no update needed
        if current_satisfied == is_feasible {
            return Ok(());
        }

        // Compute gradient for constraint refinement
        // For a·x <= b, we adjust 'a' and 'b' to better fit the data
        let violation = self.constraint.violation(sample_slice);

        // Update using perceptron-like rule
        let update_scale = if is_feasible {
            // Sample should be feasible but is violated: loosen constraint
            self.learning_rate * violation
        } else {
            // Sample should be infeasible but is satisfied: tighten constraint
            -self.learning_rate
        };

        // Create updated constraint (simplified - in practice would update coefficients)
        // This is a placeholder for demonstration
        let _ = update_scale; // TODO: implement actual coefficient updates

        Ok(())
    }

    /// Get the current learned constraint
    pub fn get_constraint(&self) -> &LinearConstraint {
        &self.constraint
    }

    /// Get number of updates performed
    pub fn update_count(&self) -> usize {
        self.update_count
    }

    /// Evaluate confidence in current constraint
    pub fn confidence(&self) -> f32 {
        if self.data_buffer.is_empty() {
            return 0.0;
        }

        // Compute accuracy on buffered data
        let correct = self
            .data_buffer
            .iter()
            .filter(|(sample, is_feasible)| {
                let satisfied = self.constraint.check(sample.as_slice().unwrap_or(&[]));
                satisfied == *is_feasible
            })
            .count();

        correct as f32 / self.data_buffer.len() as f32
    }
}

/// Anomaly detector for discovering new constraints
#[derive(Debug, Clone)]
pub struct AnomalyBasedConstraintDiscovery {
    /// Historical normal samples
    normal_samples: VecDeque<Array1<f32>>,
    /// Maximum number of normal samples to keep
    max_samples: usize,
    /// Anomaly threshold (number of standard deviations)
    anomaly_threshold: f32,
    /// Discovered constraints
    discovered_constraints: Vec<LinearConstraint>,
}

impl AnomalyBasedConstraintDiscovery {
    /// Create a new anomaly-based constraint discoverer
    pub fn new(max_samples: usize, anomaly_threshold: f32) -> Self {
        Self {
            normal_samples: VecDeque::new(),
            max_samples,
            anomaly_threshold,
            discovered_constraints: Vec::new(),
        }
    }

    /// Add a normal sample for baseline estimation
    pub fn add_normal_sample(&mut self, sample: Array1<f32>) {
        self.normal_samples.push_back(sample);
        if self.normal_samples.len() > self.max_samples {
            self.normal_samples.pop_front();
        }
    }

    /// Check if a sample is anomalous and potentially discover new constraint
    pub fn detect_anomaly(&mut self, sample: &Array1<f32>) -> bool {
        if self.normal_samples.len() < 2 {
            return false; // Not enough data for detection
        }

        // Compute statistics of normal samples
        let dim = sample.len();
        let n = self.normal_samples.len();

        // Compute mean and std dev for each dimension
        let mut is_anomalous = false;

        for d in 0..dim {
            let mean: f32 = self.normal_samples.iter().map(|s| s[d]).sum::<f32>() / n as f32;

            let variance: f32 = self
                .normal_samples
                .iter()
                .map(|s| (s[d] - mean).powi(2))
                .sum::<f32>()
                / n as f32;

            let std_dev = variance.sqrt();

            // Check if sample is outside threshold
            let z_score = (sample[d] - mean).abs() / (std_dev + 1e-8);
            if z_score > self.anomaly_threshold {
                is_anomalous = true;

                // Discover constraint: x[d] should be within [mean - threshold*std, mean + threshold*std]
                self.discover_bound_constraint(d, mean, std_dev);
            }
        }

        is_anomalous
    }

    /// Discover a bound constraint for a dimension
    fn discover_bound_constraint(&mut self, dim: usize, mean: f32, std_dev: f32) {
        // Create constraint: x[dim] <= mean + threshold * std_dev
        let upper_bound = mean + self.anomaly_threshold * std_dev;
        let mut coeffs = vec![0.0; dim + 1];
        coeffs[dim] = 1.0;

        let constraint = LinearConstraint::less_eq(coeffs, upper_bound);

        // Check if we already have a similar constraint
        let is_duplicate = self.discovered_constraints.iter().any(|c| {
            c.coefficients().len() == constraint.coefficients().len()
                && c.coefficients()
                    .iter()
                    .zip(constraint.coefficients().iter())
                    .all(|(a, b)| (a - b).abs() < 0.1)
        });

        if !is_duplicate {
            self.discovered_constraints.push(constraint);
        }
    }

    /// Get discovered constraints
    pub fn discovered_constraints(&self) -> &[LinearConstraint] {
        &self.discovered_constraints
    }

    /// Get number of discovered constraints
    pub fn num_discovered(&self) -> usize {
        self.discovered_constraints.len()
    }
}

/// Active learner for exploring constraint boundaries
#[derive(Debug, Clone)]
pub struct ActiveConstraintBoundaryLearner {
    /// Current constraint estimate
    constraint: LinearConstraint,
    /// Samples near the boundary (uncertain region)
    boundary_samples: Vec<(Array1<f32>, Option<bool>)>, // (sample, label if known)
    /// Uncertainty threshold for boundary detection
    uncertainty_threshold: f32,
    /// Maximum number of boundary samples to track
    max_boundary_samples: usize,
}

impl ActiveConstraintBoundaryLearner {
    /// Create a new active boundary learner
    pub fn new(
        initial_constraint: LinearConstraint,
        uncertainty_threshold: f32,
        max_boundary_samples: usize,
    ) -> Self {
        Self {
            constraint: initial_constraint,
            boundary_samples: Vec::new(),
            uncertainty_threshold,
            max_boundary_samples,
        }
    }

    /// Get the most informative sample to query (closest to boundary)
    pub fn query_next(&self) -> Option<Array1<f32>> {
        // Find unlabeled sample closest to decision boundary
        self.boundary_samples
            .iter()
            .filter(|(_, label)| label.is_none())
            .min_by(|(s1, _), (s2, _)| {
                let v1 = self
                    .constraint
                    .violation(s1.as_slice().unwrap_or(&[]))
                    .abs();
                let v2 = self
                    .constraint
                    .violation(s2.as_slice().unwrap_or(&[]))
                    .abs();
                v1.partial_cmp(&v2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(s, _)| s.clone())
    }

    /// Add a labeled sample
    pub fn add_labeled_sample(&mut self, sample: Array1<f32>, is_feasible: bool) {
        let violation = self
            .constraint
            .violation(sample.as_slice().unwrap_or(&[]))
            .abs();

        // Add to boundary samples if near boundary
        if violation < self.uncertainty_threshold {
            self.boundary_samples.push((sample, Some(is_feasible)));
            if self.boundary_samples.len() > self.max_boundary_samples {
                self.boundary_samples.remove(0);
            }
        }
    }

    /// Add an unlabeled sample for potential querying
    pub fn add_unlabeled_sample(&mut self, sample: Array1<f32>) {
        let violation = self
            .constraint
            .violation(sample.as_slice().unwrap_or(&[]))
            .abs();

        if violation < self.uncertainty_threshold {
            self.boundary_samples.push((sample, None));
            if self.boundary_samples.len() > self.max_boundary_samples {
                self.boundary_samples.remove(0);
            }
        }
    }

    /// Refine constraint based on labeled boundary samples
    pub fn refine(&mut self) -> LogicResult<()> {
        // Use labeled boundary samples to refine constraint
        // This is a simplified version - in practice would use SVM or similar
        let labeled: Vec<_> = self
            .boundary_samples
            .iter()
            .filter_map(|(s, l)| l.map(|label| (s, label)))
            .collect();

        if labeled.len() < 2 {
            return Ok(()); // Not enough data
        }

        // Placeholder for actual refinement logic
        // Would typically use margin-based learning or similar
        Ok(())
    }

    /// Get the current constraint
    pub fn get_constraint(&self) -> &LinearConstraint {
        &self.constraint
    }

    /// Get number of boundary samples
    pub fn num_boundary_samples(&self) -> usize {
        self.boundary_samples.len()
    }

    /// Get number of unlabeled samples
    pub fn num_unlabeled(&self) -> usize {
        self.boundary_samples
            .iter()
            .filter(|(_, l)| l.is_none())
            .count()
    }
}

/// Feedback-based constraint tuner
#[derive(Debug, Clone)]
pub struct FeedbackConstraintTuner {
    /// Current constraint
    constraint: LinearConstraint,
    /// Feedback history (violation amount, user satisfaction)
    feedback_history: Vec<(f32, f32)>, // (violation, satisfaction in [0, 1])
    /// Adaptation rate
    #[allow(dead_code)]
    adaptation_rate: f32,
    /// Target satisfaction level
    target_satisfaction: f32,
}

impl FeedbackConstraintTuner {
    /// Create a new feedback-based tuner
    pub fn new(
        initial_constraint: LinearConstraint,
        adaptation_rate: f32,
        target_satisfaction: f32,
    ) -> Self {
        Self {
            constraint: initial_constraint,
            feedback_history: Vec::new(),
            adaptation_rate,
            target_satisfaction,
        }
    }

    /// Add user feedback for a sample
    pub fn add_feedback(&mut self, sample: &Array1<f32>, satisfaction: f32) -> LogicResult<()> {
        let violation = self.constraint.violation(sample.as_slice().unwrap_or(&[]));
        self.feedback_history.push((violation, satisfaction));

        // Tune constraint based on feedback
        self.tune()?;

        Ok(())
    }

    /// Tune constraint based on accumulated feedback
    fn tune(&mut self) -> LogicResult<()> {
        if self.feedback_history.len() < 5 {
            return Ok(()); // Need more data
        }

        // Compute average satisfaction
        let avg_satisfaction: f32 = self.feedback_history.iter().map(|(_, s)| s).sum::<f32>()
            / self.feedback_history.len() as f32;

        // If satisfaction is below target, adjust constraint
        let satisfaction_gap = self.target_satisfaction - avg_satisfaction;

        if satisfaction_gap.abs() > 0.1 {
            // Significant gap - adjust constraint tightness
            // Positive gap means we need to be less strict
            // This is a placeholder for actual tuning logic
            let _ = satisfaction_gap; // TODO: implement actual tuning
        }

        Ok(())
    }

    /// Get the current constraint
    pub fn get_constraint(&self) -> &LinearConstraint {
        &self.constraint
    }

    /// Get average satisfaction from recent feedback
    pub fn average_satisfaction(&self) -> f32 {
        if self.feedback_history.is_empty() {
            return 0.0;
        }

        self.feedback_history.iter().map(|(_, s)| s).sum::<f32>()
            / self.feedback_history.len() as f32
    }

    /// Get number of feedback samples
    pub fn num_feedback_samples(&self) -> usize {
        self.feedback_history.len()
    }
}

/// Unified online learning system combining multiple strategies
#[derive(Debug, Clone)]
pub struct OnlineLearningSystem {
    /// Incremental learner
    incremental_learner: OnlineConstraintLearner,
    /// Anomaly detector
    anomaly_detector: AnomalyBasedConstraintDiscovery,
    /// Active learner
    active_learner: ActiveConstraintBoundaryLearner,
    /// Feedback tuner
    feedback_tuner: FeedbackConstraintTuner,
    /// Enable/disable each component
    use_incremental: bool,
    use_anomaly: bool,
    use_active: bool,
    use_feedback: bool,
}

impl OnlineLearningSystem {
    /// Create a new online learning system
    pub fn new(initial_constraint: LinearConstraint) -> Self {
        Self {
            incremental_learner: OnlineConstraintLearner::new(
                initial_constraint.clone(),
                0.01,
                1000,
            ),
            anomaly_detector: AnomalyBasedConstraintDiscovery::new(1000, 3.0),
            active_learner: ActiveConstraintBoundaryLearner::new(
                initial_constraint.clone(),
                0.1,
                100,
            ),
            feedback_tuner: FeedbackConstraintTuner::new(initial_constraint, 0.01, 0.8),
            use_incremental: true,
            use_anomaly: true,
            use_active: true,
            use_feedback: true,
        }
    }

    /// Process a new labeled sample
    pub fn process_labeled_sample(
        &mut self,
        sample: Array1<f32>,
        is_feasible: bool,
    ) -> LogicResult<()> {
        if self.use_incremental {
            self.incremental_learner
                .observe(sample.clone(), is_feasible)?;
        }

        if self.use_active {
            self.active_learner
                .add_labeled_sample(sample.clone(), is_feasible);
        }

        if is_feasible && self.use_anomaly {
            self.anomaly_detector.add_normal_sample(sample);
        }

        Ok(())
    }

    /// Process an unlabeled sample (for anomaly detection and active learning)
    pub fn process_unlabeled_sample(&mut self, sample: Array1<f32>) {
        if self.use_anomaly {
            self.anomaly_detector.detect_anomaly(&sample);
        }

        if self.use_active {
            self.active_learner.add_unlabeled_sample(sample);
        }
    }

    /// Add user feedback
    pub fn add_feedback(&mut self, sample: &Array1<f32>, satisfaction: f32) -> LogicResult<()> {
        if self.use_feedback {
            self.feedback_tuner.add_feedback(sample, satisfaction)?;
        }
        Ok(())
    }

    /// Get the current best constraint estimate
    pub fn get_best_constraint(&self) -> &LinearConstraint {
        // Use the incremental learner's constraint as primary
        // Could be enhanced to ensemble multiple learned constraints
        self.incremental_learner.get_constraint()
    }

    /// Get confidence in current constraint
    pub fn confidence(&self) -> f32 {
        self.incremental_learner.confidence()
    }

    /// Get discovered anomaly-based constraints
    pub fn discovered_constraints(&self) -> &[LinearConstraint] {
        self.anomaly_detector.discovered_constraints()
    }

    /// Get next sample to query (active learning)
    pub fn query_next(&self) -> Option<Array1<f32>> {
        if self.use_active {
            self.active_learner.query_next()
        } else {
            None
        }
    }

    /// Enable/disable components
    pub fn set_use_incremental(&mut self, use_it: bool) {
        self.use_incremental = use_it;
    }

    pub fn set_use_anomaly(&mut self, use_it: bool) {
        self.use_anomaly = use_it;
    }

    pub fn set_use_active(&mut self, use_it: bool) {
        self.use_active = use_it;
    }

    pub fn set_use_feedback(&mut self, use_it: bool) {
        self.use_feedback = use_it;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_learner_basic() -> LogicResult<()> {
        let constraint = LinearConstraint::less_eq(vec![1.0], 5.0);
        let mut learner = OnlineConstraintLearner::new(constraint, 0.1, 100);

        // Add some observations
        learner.observe(Array1::from_vec(vec![3.0]), true)?; // Feasible
        learner.observe(Array1::from_vec(vec![7.0]), false)?; // Infeasible

        assert_eq!(learner.update_count(), 2);
        assert!(learner.confidence() > 0.0);

        Ok(())
    }

    #[test]
    fn test_anomaly_detection() {
        let mut detector = AnomalyBasedConstraintDiscovery::new(100, 3.0);

        // Add normal samples
        for _ in 0..20 {
            detector.add_normal_sample(Array1::from_vec(vec![5.0, 10.0]));
        }

        // Detect anomaly
        let is_anomaly = detector.detect_anomaly(&Array1::from_vec(vec![50.0, 100.0]));
        assert!(is_anomaly);
    }

    #[test]
    fn test_active_learning() {
        let constraint = LinearConstraint::less_eq(vec![1.0], 5.0);
        let mut learner = ActiveConstraintBoundaryLearner::new(constraint, 1.0, 100);

        learner.add_unlabeled_sample(Array1::from_vec(vec![4.9])); // Near boundary
        learner.add_unlabeled_sample(Array1::from_vec(vec![10.0])); // Far from boundary

        assert_eq!(learner.num_unlabeled(), 1); // Only near-boundary sample added
    }

    #[test]
    fn test_feedback_tuner() -> LogicResult<()> {
        let constraint = LinearConstraint::less_eq(vec![1.0], 5.0);
        let mut tuner = FeedbackConstraintTuner::new(constraint, 0.1, 0.8);

        tuner.add_feedback(&Array1::from_vec(vec![3.0]), 0.9)?; // High satisfaction
        tuner.add_feedback(&Array1::from_vec(vec![4.0]), 0.7)?; // Medium satisfaction

        assert_eq!(tuner.num_feedback_samples(), 2);
        assert!(tuner.average_satisfaction() > 0.0);

        Ok(())
    }

    #[test]
    fn test_online_learning_system() -> LogicResult<()> {
        let constraint = LinearConstraint::less_eq(vec![1.0], 5.0);
        let mut system = OnlineLearningSystem::new(constraint);

        system.process_labeled_sample(Array1::from_vec(vec![3.0]), true)?;
        system.process_unlabeled_sample(Array1::from_vec(vec![4.5]));
        system.add_feedback(&Array1::from_vec(vec![3.5]), 0.9)?;

        assert!(system.confidence() > 0.0);

        Ok(())
    }
}
