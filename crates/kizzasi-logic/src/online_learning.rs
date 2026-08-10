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
        // Positive update_scale → loosen (raise b); negative → tighten (lower b)
        let update_scale = if is_feasible {
            // Sample should be feasible but is violated: loosen constraint
            self.learning_rate * violation
        } else {
            // Sample should be infeasible but is satisfied: tighten constraint
            -self.learning_rate
        };

        self.constraint.shift_rhs(update_scale);
        self.constraint.update_coefficients_towards(
            sample_slice,
            update_scale.signum() * self.learning_rate * 0.1,
        );

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

    /// Refine the linear constraint boundary using a multi-pass margin-based
    /// perceptron update over all labeled boundary samples.
    ///
    /// For each training epoch the learning rate is annealed as
    /// `lr = base_lr / (1 + epoch)`.  On every mis-classified sample:
    ///
    /// - **Feasible but predicted violated** → loosen the boundary by shifting
    ///   the RHS upward and nudging the coefficient vector away from the sample.
    /// - **Infeasible but predicted satisfied** → tighten the boundary by
    ///   shifting the RHS downward and nudging the coefficient vector towards
    ///   the sample.
    ///
    /// At least two labeled samples are required; fewer data points return
    /// immediately without modifying the constraint.
    pub fn refine(&mut self) -> LogicResult<()> {
        let labeled: Vec<_> = self
            .boundary_samples
            .iter()
            .filter_map(|(s, l)| l.map(|label| (s.clone(), label)))
            .collect();

        if labeled.len() < 2 {
            return Ok(());
        }

        let max_epochs = 5_usize;
        let base_lr = 0.1_f32;

        for epoch in 0..max_epochs {
            let lr = base_lr / (1.0 + epoch as f32);
            for (sample, is_feasible) in &labeled {
                let sample_slice = sample.as_slice().unwrap_or(&[]);
                let predicted_feasible = self.constraint.check(sample_slice);
                if predicted_feasible == *is_feasible {
                    continue; // Correctly classified — no update needed
                }
                // Mis-classified: nudge the boundary
                let violation = self.constraint.violation(sample_slice);
                let margin = violation.max(lr); // At least lr to guarantee movement
                if *is_feasible {
                    // Sample should be feasible but constraint says violated → loosen boundary
                    self.constraint.shift_rhs(margin * lr);
                    self.constraint
                        .update_coefficients_towards(sample_slice, -lr * 0.1);
                } else {
                    // Sample should be infeasible but constraint says satisfied → tighten boundary
                    self.constraint.shift_rhs(-margin * lr);
                    self.constraint
                        .update_coefficients_towards(sample_slice, lr * 0.1);
                }
            }
        }
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
        // Positive gap: satisfaction below target → loosen constraint (raise b)
        // Negative gap: satisfaction above target → tighten constraint (lower b)
        let satisfaction_gap = self.target_satisfaction - avg_satisfaction;

        if satisfaction_gap.abs() > 0.1 {
            self.constraint
                .shift_rhs(satisfaction_gap * self.adaptation_rate);
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

    // -------------------------------------------------------------------------
    // Track 1 tests: OnlineConstraintLearner::refine_constraint
    // -------------------------------------------------------------------------

    #[test]
    fn test_online_learner_refine_loosens_constraint() -> LogicResult<()> {
        // Constraint: x[0] ≤ 3.0.
        // Feed a sample at x=4.0 labeled feasible.  The constraint predicts
        // violated (4.0 > 3.0), but the label says feasible → mismatch →
        // update_scale is positive → shift_rhs raises b above 3.0.
        let constraint = LinearConstraint::less_eq(vec![1.0], 3.0);
        let mut learner = OnlineConstraintLearner::new(constraint, 0.1, 10);
        let rhs_before = learner.get_constraint().rhs();
        learner.observe(Array1::from(vec![4.0_f32]), true)?;
        assert!(
            learner.get_constraint().rhs() > rhs_before,
            "feasible-but-violated sample should loosen constraint: {} → {}",
            rhs_before,
            learner.get_constraint().rhs()
        );
        Ok(())
    }

    #[test]
    fn test_online_learner_refine_tightens_constraint() -> LogicResult<()> {
        // Constraint: x[0] ≤ 3.0.
        // Feed a sample at x=1.0 labeled infeasible.  The constraint predicts
        // satisfied (1.0 ≤ 3.0), but the label says infeasible → mismatch →
        // update_scale is negative → shift_rhs lowers b below 3.0.
        let constraint = LinearConstraint::less_eq(vec![1.0], 3.0);
        let mut learner = OnlineConstraintLearner::new(constraint, 0.1, 10);
        let rhs_before = learner.get_constraint().rhs();
        learner.observe(Array1::from(vec![1.0_f32]), false)?;
        assert!(
            learner.get_constraint().rhs() < rhs_before,
            "infeasible-but-satisfied sample should tighten constraint: {} → {}",
            rhs_before,
            learner.get_constraint().rhs()
        );
        Ok(())
    }

    #[test]
    fn test_online_learner_refine_no_update_when_correct() -> LogicResult<()> {
        // Constraint: x[0] ≤ 3.0.
        // Feed a sample at x=1.0 labeled feasible.  The constraint predicts
        // satisfied (1.0 ≤ 3.0) and label agrees → no update → rhs stays 3.0.
        let constraint = LinearConstraint::less_eq(vec![1.0], 3.0);
        let mut learner = OnlineConstraintLearner::new(constraint, 0.1, 10);
        let rhs_before = learner.get_constraint().rhs();
        learner.observe(Array1::from(vec![1.0_f32]), true)?;
        assert_eq!(learner.update_count(), 1);
        assert!(
            (learner.get_constraint().rhs() - rhs_before).abs() < f32::EPSILON,
            "correctly-classified sample must not change rhs: {} → {}",
            rhs_before,
            learner.get_constraint().rhs()
        );
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Track 1 tests: FeedbackConstraintTuner::tune
    // -------------------------------------------------------------------------

    #[test]
    fn test_feedback_tuner_loosens_when_satisfaction_below_target() -> LogicResult<()> {
        // target=0.9, adaptation_rate=0.5, initial rhs=5.0.
        // Feed 5 feedbacks with satisfaction=0.0 → avg=0.0 → gap=0.9 > 0.1 →
        // shift_rhs(0.9 * 0.5 = 0.45) → rhs becomes 5.45.
        let constraint = LinearConstraint::less_eq(vec![1.0], 5.0);
        let mut tuner = FeedbackConstraintTuner::new(constraint, 0.5, 0.9);
        let rhs_before = tuner.get_constraint().rhs();
        for _ in 0..5 {
            tuner.add_feedback(&Array1::from(vec![2.0_f32]), 0.0)?;
        }
        assert!(
            tuner.get_constraint().rhs() > rhs_before,
            "satisfaction below target should loosen constraint: {} → {}",
            rhs_before,
            tuner.get_constraint().rhs()
        );
        Ok(())
    }

    #[test]
    fn test_feedback_tuner_no_change_below_threshold() -> LogicResult<()> {
        // target=0.9, adaptation_rate=0.5, initial rhs=5.0.
        // Feed 5 feedbacks with satisfaction=0.85 → avg=0.85 → gap=0.05 < 0.1 →
        // no shift → rhs stays exactly 5.0.
        let constraint = LinearConstraint::less_eq(vec![1.0], 5.0);
        let mut tuner = FeedbackConstraintTuner::new(constraint, 0.5, 0.9);
        let rhs_before = tuner.get_constraint().rhs();
        for _ in 0..5 {
            tuner.add_feedback(&Array1::from(vec![2.0_f32]), 0.85)?;
        }
        assert!(
            (tuner.get_constraint().rhs() - rhs_before).abs() < f32::EPSILON,
            "gap below threshold must not change rhs: {} → {}",
            rhs_before,
            tuner.get_constraint().rhs()
        );
        Ok(())
    }
}
