//! Signal Temporal Logic (STL) with Quantitative Robustness Semantics
//!
//! STL extends Linear Temporal Logic (LTL) for continuous-time signals with:
//! - Time-bounded temporal operators (e.g., Eventually[0,5] φ)
//! - Quantitative semantics: robustness degree ρ (how much a signal satisfies φ)
//! - Continuous predicates: μ(x,t) instead of boolean predicates
//!
//! # Robustness Semantics
//!
//! - ρ(φ, x, t) > 0: φ is satisfied at time t with margin ρ
//! - ρ(φ, x, t) < 0: φ is violated at time t with margin |ρ|
//! - ρ(φ, x, t) = 0: φ is on the boundary
//!
//! # Applications
//!
//! - Continuous-time control verification
//! - Time-series constraint specification
//! - Robust temporal property checking
//! - Safety-critical system monitoring

use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Time interval for temporal operators
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeInterval {
    /// Lower bound (inclusive)
    pub lower: f32,
    /// Upper bound (inclusive)
    pub upper: f32,
}

impl TimeInterval {
    /// Create a new time interval
    pub fn new(lower: f32, upper: f32) -> Self {
        assert!(
            lower >= 0.0 && upper >= lower,
            "Invalid time interval: [{}, {}]",
            lower,
            upper
        );
        Self { lower, upper }
    }

    /// Unbounded interval [0, ∞)
    pub fn unbounded() -> Self {
        Self {
            lower: 0.0,
            upper: f32::INFINITY,
        }
    }

    /// Check if a time value is in this interval
    pub fn contains(&self, t: f32) -> bool {
        t >= self.lower && t <= self.upper
    }

    /// Duration of the interval
    pub fn duration(&self) -> f32 {
        self.upper - self.lower
    }
}

/// STL Formula with quantitative semantics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum STLFormula {
    /// Atomic predicate: μ(x) ≥ 0
    Predicate {
        /// Name of the predicate
        name: String,
        /// Predicate function: returns robustness value
        /// Stored as dimension index and threshold for serialization
        dimension: usize,
        threshold: f32,
        /// If true, checks x[dim] ≥ threshold, else x[dim] ≤ threshold
        greater_than: bool,
    },

    /// Negation: ¬φ
    /// ρ(¬φ) = -ρ(φ)
    Not(Box<STLFormula>),

    /// Conjunction: φ₁ ∧ φ₂
    /// ρ(φ₁ ∧ φ₂) = min(ρ(φ₁), ρ(φ₂))
    And(Box<STLFormula>, Box<STLFormula>),

    /// Disjunction: φ₁ ∨ φ₂
    /// ρ(φ₁ ∨ φ₂) = max(ρ(φ₁), ρ(φ₂))
    Or(Box<STLFormula>, Box<STLFormula>),

    /// Implies: φ₁ → φ₂
    /// Equivalent to ¬φ₁ ∨ φ₂
    Implies(Box<STLFormula>, Box<STLFormula>),

    /// Eventually (Future): ◇[a,b] φ
    /// ρ(◇[a,b] φ, t) = sup_{t' ∈ [t+a, t+b]} ρ(φ, t')
    Eventually {
        interval: TimeInterval,
        formula: Box<STLFormula>,
    },

    /// Always (Globally): □[a,b] φ
    /// ρ(□[a,b] φ, t) = inf_{t' ∈ [t+a, t+b]} ρ(φ, t')
    Always {
        interval: TimeInterval,
        formula: Box<STLFormula>,
    },

    /// Until: φ₁ U[a,b] φ₂
    /// φ₁ must hold until φ₂ becomes true within [a,b]
    Until {
        interval: TimeInterval,
        lhs: Box<STLFormula>,
        rhs: Box<STLFormula>,
    },

    /// Release: φ₁ R[a,b] φ₂
    /// φ₂ must hold until φ₁ becomes true (dual of Until)
    Release {
        interval: TimeInterval,
        lhs: Box<STLFormula>,
        rhs: Box<STLFormula>,
    },
}

impl STLFormula {
    /// Create a predicate: x[dim] ≥ threshold
    pub fn greater_eq(name: impl Into<String>, dimension: usize, threshold: f32) -> Self {
        Self::Predicate {
            name: name.into(),
            dimension,
            threshold,
            greater_than: true,
        }
    }

    /// Create a predicate: x[dim] ≤ threshold
    pub fn less_eq(name: impl Into<String>, dimension: usize, threshold: f32) -> Self {
        Self::Predicate {
            name: name.into(),
            dimension,
            threshold,
            greater_than: false,
        }
    }

    /// Create conjunction
    pub fn and(lhs: STLFormula, rhs: STLFormula) -> Self {
        Self::And(Box::new(lhs), Box::new(rhs))
    }

    /// Create disjunction
    pub fn or(lhs: STLFormula, rhs: STLFormula) -> Self {
        Self::Or(Box::new(lhs), Box::new(rhs))
    }

    /// Create implication
    pub fn implies(lhs: STLFormula, rhs: STLFormula) -> Self {
        Self::Implies(Box::new(lhs), Box::new(rhs))
    }

    /// Create eventually with time bounds
    pub fn eventually(interval: TimeInterval, formula: STLFormula) -> Self {
        Self::Eventually {
            interval,
            formula: Box::new(formula),
        }
    }

    /// Create always with time bounds
    pub fn always(interval: TimeInterval, formula: STLFormula) -> Self {
        Self::Always {
            interval,
            formula: Box::new(formula),
        }
    }

    /// Create until with time bounds
    pub fn until(interval: TimeInterval, lhs: STLFormula, rhs: STLFormula) -> Self {
        Self::Until {
            interval,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    /// Create release with time bounds
    pub fn release(interval: TimeInterval, lhs: STLFormula, rhs: STLFormula) -> Self {
        Self::Release {
            interval,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    /// Compute quantitative robustness at a point
    pub fn robustness(&self, x: &[f32]) -> f32 {
        match self {
            Self::Predicate {
                dimension,
                threshold,
                greater_than,
                ..
            } => {
                if *dimension >= x.len() {
                    return f32::NEG_INFINITY;
                }
                let value = x[*dimension];
                if *greater_than {
                    value - threshold // ρ = x - threshold (positive if satisfied)
                } else {
                    threshold - value // ρ = threshold - x (positive if satisfied)
                }
            }
            Self::Not(phi) => -phi.robustness(x),
            Self::And(phi1, phi2) => phi1.robustness(x).min(phi2.robustness(x)),
            Self::Or(phi1, phi2) => phi1.robustness(x).max(phi2.robustness(x)),
            Self::Implies(phi1, phi2) => {
                // φ₁ → φ₂ ≡ ¬φ₁ ∨ φ₂
                (-phi1.robustness(x)).max(phi2.robustness(x))
            }
            // For temporal operators on single point, evaluate at current time
            Self::Eventually { formula, .. } => formula.robustness(x),
            Self::Always { formula, .. } => formula.robustness(x),
            Self::Until { rhs, .. } => rhs.robustness(x),
            Self::Release { rhs, .. } => rhs.robustness(x),
        }
    }

    /// Check if formula is satisfied (robustness ≥ 0)
    pub fn check(&self, x: &[f32]) -> bool {
        self.robustness(x) >= 0.0
    }

    /// Get the minimum horizon (maximum time needed to evaluate)
    pub fn horizon(&self) -> f32 {
        match self {
            Self::Predicate { .. } => 0.0,
            Self::Not(phi) => phi.horizon(),
            Self::And(phi1, phi2) | Self::Or(phi1, phi2) | Self::Implies(phi1, phi2) => {
                phi1.horizon().max(phi2.horizon())
            }
            Self::Eventually { interval, formula } | Self::Always { interval, formula } => {
                interval.upper + formula.horizon()
            }
            Self::Until { interval, lhs, rhs } | Self::Release { interval, lhs, rhs } => {
                interval.upper + lhs.horizon().max(rhs.horizon())
            }
        }
    }
}

/// Implement the Not trait for STLFormula to support the ! operator
impl std::ops::Not for STLFormula {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self::Not(Box::new(self))
    }
}

/// Signal: time series of multi-dimensional values
#[derive(Debug, Clone)]
pub struct Signal {
    /// Time stamps (must be sorted)
    pub times: Vec<f32>,
    /// Values at each time stamp
    pub values: Vec<Array1<f32>>,
}

impl Signal {
    /// Create a new signal
    pub fn new(times: Vec<f32>, values: Vec<Array1<f32>>) -> Self {
        assert_eq!(
            times.len(),
            values.len(),
            "Time and value vectors must have same length"
        );
        assert!(
            times.windows(2).all(|w| w[0] <= w[1]),
            "Times must be sorted"
        );
        Self { times, values }
    }

    /// Get value at specific time (linear interpolation)
    pub fn at(&self, t: f32) -> Option<Array1<f32>> {
        if self.times.is_empty() {
            return None;
        }

        // Handle boundary cases
        if t <= self.times[0] {
            return Some(self.values[0].clone());
        }
        if t >= *self.times.last().unwrap() {
            return Some(self.values.last().unwrap().clone());
        }

        // Binary search for interval
        let idx = self
            .times
            .binary_search_by(|probe| probe.partial_cmp(&t).unwrap())
            .unwrap_or_else(|i| i);

        if idx == 0 {
            return Some(self.values[0].clone());
        }

        // Linear interpolation
        let t0 = self.times[idx - 1];
        let t1 = self.times[idx];
        let v0 = &self.values[idx - 1];
        let v1 = &self.values[idx];

        let alpha = (t - t0) / (t1 - t0);
        Some(v0 * (1.0 - alpha) + v1 * alpha)
    }

    /// Get time window
    pub fn time_range(&self) -> (f32, f32) {
        if self.times.is_empty() {
            (0.0, 0.0)
        } else {
            (self.times[0], *self.times.last().unwrap())
        }
    }

    /// Number of samples
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Check if signal is empty
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }
}

/// STL Monitor: computes robustness of signal traces
pub struct STLMonitor {
    /// Formula to monitor
    formula: STLFormula,
    /// Time resolution for discrete monitoring
    #[allow(dead_code)]
    dt: f32,
}

impl STLMonitor {
    /// Create a new STL monitor
    pub fn new(formula: STLFormula, dt: f32) -> Self {
        assert!(dt > 0.0, "Time resolution must be positive");
        Self { formula, dt }
    }

    /// Compute robustness of a signal over time
    ///
    /// Returns vector of (time, robustness) pairs
    pub fn monitor(&self, signal: &Signal) -> Vec<(f32, f32)> {
        if signal.is_empty() {
            return vec![];
        }

        let mut results = Vec::new();

        // Evaluate formula at each time step
        for (i, &t) in signal.times.iter().enumerate() {
            let robustness = self.evaluate_at_time(signal, i, t);
            results.push((t, robustness));
        }

        results
    }

    /// Evaluate formula at specific time index
    fn evaluate_at_time(&self, signal: &Signal, time_idx: usize, current_time: f32) -> f32 {
        Self::evaluate_formula(&self.formula, signal, time_idx, current_time)
    }

    /// Recursive evaluation of STL formula
    fn evaluate_formula(
        formula: &STLFormula,
        signal: &Signal,
        time_idx: usize,
        current_time: f32,
    ) -> f32 {
        match formula {
            STLFormula::Predicate {
                dimension,
                threshold,
                greater_than,
                ..
            } => {
                let x = &signal.values[time_idx];
                if *dimension >= x.len() {
                    return f32::NEG_INFINITY;
                }
                let value = x[*dimension];
                if *greater_than {
                    value - threshold
                } else {
                    threshold - value
                }
            }

            STLFormula::Not(phi) => -Self::evaluate_formula(phi, signal, time_idx, current_time),

            STLFormula::And(phi1, phi2) => {
                let r1 = Self::evaluate_formula(phi1, signal, time_idx, current_time);
                let r2 = Self::evaluate_formula(phi2, signal, time_idx, current_time);
                r1.min(r2)
            }

            STLFormula::Or(phi1, phi2) => {
                let r1 = Self::evaluate_formula(phi1, signal, time_idx, current_time);
                let r2 = Self::evaluate_formula(phi2, signal, time_idx, current_time);
                r1.max(r2)
            }

            STLFormula::Implies(phi1, phi2) => {
                let r1 = Self::evaluate_formula(phi1, signal, time_idx, current_time);
                let r2 = Self::evaluate_formula(phi2, signal, time_idx, current_time);
                (-r1).max(r2)
            }

            STLFormula::Eventually { interval, formula } => {
                let mut max_robustness = f32::NEG_INFINITY;

                // Find all time indices in [t + a, t + b]
                let t_start = current_time + interval.lower;
                let t_end = current_time + interval.upper;

                for (idx, &t) in signal.times.iter().enumerate() {
                    if t >= t_start && t <= t_end {
                        let rob = Self::evaluate_formula(formula, signal, idx, t);
                        max_robustness = max_robustness.max(rob);
                    }
                }

                max_robustness
            }

            STLFormula::Always { interval, formula } => {
                let mut min_robustness = f32::INFINITY;

                // Find all time indices in [t + a, t + b]
                let t_start = current_time + interval.lower;
                let t_end = current_time + interval.upper;

                for (idx, &t) in signal.times.iter().enumerate() {
                    if t >= t_start && t <= t_end {
                        let rob = Self::evaluate_formula(formula, signal, idx, t);
                        min_robustness = min_robustness.min(rob);
                    }
                }

                min_robustness
            }

            STLFormula::Until { interval, lhs, rhs } => {
                let mut max_robustness = f32::NEG_INFINITY;

                let t_start = current_time + interval.lower;
                let t_end = current_time + interval.upper;

                for (idx, &t) in signal.times.iter().enumerate() {
                    if t >= t_start && t <= t_end {
                        // Robustness at this potential satisfaction point
                        let rob_rhs = Self::evaluate_formula(rhs, signal, idx, t);

                        // Check that lhs holds until this point
                        let mut min_lhs = f32::INFINITY;
                        for (prev_idx, &prev_t) in signal.times.iter().enumerate() {
                            if prev_t >= current_time && prev_t < t {
                                let rob_lhs = Self::evaluate_formula(lhs, signal, prev_idx, prev_t);
                                min_lhs = min_lhs.min(rob_lhs);
                            }
                        }

                        max_robustness = max_robustness.max(min_lhs.min(rob_rhs));
                    }
                }

                max_robustness
            }

            STLFormula::Release { interval, lhs, rhs } => {
                // φ₁ R φ₂: φ₂ holds until φ₁ (dual of Until)
                // Compute using Until: ¬(¬φ₁ U ¬φ₂)
                let neg_lhs = STLFormula::Not(Box::new((**lhs).clone()));
                let neg_rhs = STLFormula::Not(Box::new((**rhs).clone()));
                let until_formula = STLFormula::Until {
                    interval: *interval,
                    lhs: Box::new(neg_lhs),
                    rhs: Box::new(neg_rhs),
                };
                -Self::evaluate_formula(&until_formula, signal, time_idx, current_time)
            }
        }
    }

    /// Check if signal satisfies formula (minimum robustness ≥ 0)
    pub fn satisfies(&self, signal: &Signal) -> bool {
        let results = self.monitor(signal);
        results.iter().all(|(_, rob)| *rob >= 0.0)
    }

    /// Get minimum robustness over entire signal
    pub fn min_robustness(&self, signal: &Signal) -> f32 {
        let results = self.monitor(signal);
        results
            .iter()
            .map(|(_, rob)| *rob)
            .fold(f32::INFINITY, f32::min)
    }
}

/// Online STL Monitor: incrementally processes streaming data
pub struct OnlineSTLMonitor {
    /// Formula to monitor
    formula: STLFormula,
    /// Buffer of recent observations
    buffer: VecDeque<(f32, Array1<f32>)>,
    /// Maximum horizon needed
    horizon: f32,
}

impl OnlineSTLMonitor {
    /// Create a new online monitor
    pub fn new(formula: STLFormula) -> Self {
        let horizon = formula.horizon();
        Self {
            formula,
            buffer: VecDeque::new(),
            horizon,
        }
    }

    /// Add a new observation and compute robustness
    pub fn update(&mut self, time: f32, value: Array1<f32>) -> f32 {
        // Add to buffer
        self.buffer.push_back((time, value.clone()));

        // Remove old observations outside horizon
        while let Some((t, _)) = self.buffer.front() {
            if time - t > self.horizon * 2.0 {
                self.buffer.pop_front();
            } else {
                break;
            }
        }

        // Convert buffer to signal
        let signal = self.buffer_to_signal();

        // Evaluate at current time (last index)
        if signal.times.is_empty() {
            return f32::NEG_INFINITY;
        }

        let time_idx = signal.times.len() - 1;
        self.evaluate_formula(&self.formula, &signal, time_idx, time)
    }

    /// Convert buffer to signal
    fn buffer_to_signal(&self) -> Signal {
        let times: Vec<f32> = self.buffer.iter().map(|(t, _)| *t).collect();
        let values: Vec<Array1<f32>> = self.buffer.iter().map(|(_, v)| v.clone()).collect();
        Signal::new(times, values)
    }

    /// Evaluate formula (same as STLMonitor)
    fn evaluate_formula(
        &self,
        formula: &STLFormula,
        signal: &Signal,
        time_idx: usize,
        current_time: f32,
    ) -> f32 {
        // Reuse evaluation logic from STLMonitor
        STLMonitor::evaluate_formula(formula, signal, time_idx, current_time)
    }

    /// Check if current state satisfies formula
    pub fn check(&self) -> bool {
        if self.buffer.is_empty() {
            return false;
        }
        let signal = self.buffer_to_signal();
        let time_idx = signal.times.len() - 1;
        let current_time = signal.times[time_idx];
        self.evaluate_formula(&self.formula, &signal, time_idx, current_time) >= 0.0
    }

    /// Reset the monitor
    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_interval() {
        let interval = TimeInterval::new(0.0, 5.0);
        assert!(interval.contains(2.5));
        assert!(!interval.contains(6.0));
        assert_eq!(interval.duration(), 5.0);
    }

    #[test]
    fn test_predicate_robustness() {
        // x[0] ≥ 5.0
        let phi = STLFormula::greater_eq("x_geq_5", 0, 5.0);

        assert_eq!(phi.robustness(&[6.0]), 1.0); // 6.0 - 5.0 = 1.0
        assert_eq!(phi.robustness(&[5.0]), 0.0); // 5.0 - 5.0 = 0.0
        assert_eq!(phi.robustness(&[4.0]), -1.0); // 4.0 - 5.0 = -1.0

        assert!(phi.check(&[6.0]));
        assert!(phi.check(&[5.0]));
        assert!(!phi.check(&[4.0]));
    }

    #[test]
    fn test_logical_operators() {
        let phi1 = STLFormula::greater_eq("x0_geq_5", 0, 5.0);
        let phi2 = STLFormula::less_eq("x1_leq_10", 1, 10.0);

        // AND: min of robustness
        let phi_and = STLFormula::and(phi1.clone(), phi2.clone());
        assert_eq!(phi_and.robustness(&[6.0, 9.0]), 1.0_f32.min(1.0)); // min(1.0, 1.0) = 1.0

        // OR: max of robustness
        let phi_or = STLFormula::or(phi1.clone(), phi2.clone());
        assert_eq!(phi_or.robustness(&[4.0, 11.0]), (-1.0_f32).max(-1.0)); // max(-1.0, -1.0) = -1.0

        // NOT: negate robustness
        let phi_not = !phi1;
        assert_eq!(phi_not.robustness(&[6.0]), -1.0); // -(6.0 - 5.0) = -1.0
    }

    #[test]
    fn test_signal_creation() {
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![
            Array1::from_vec(vec![1.0]),
            Array1::from_vec(vec![2.0]),
            Array1::from_vec(vec![3.0]),
            Array1::from_vec(vec![4.0]),
        ];

        let signal = Signal::new(times, values);
        assert_eq!(signal.len(), 4);
        assert_eq!(signal.time_range(), (0.0, 3.0));
    }

    #[test]
    fn test_signal_interpolation() {
        let times = vec![0.0, 2.0];
        let values = vec![Array1::from_vec(vec![0.0]), Array1::from_vec(vec![10.0])];

        let signal = Signal::new(times, values);

        // At t=1.0 (midpoint), should be 5.0
        let val = signal.at(1.0).unwrap();
        assert!((val[0] - 5.0).abs() < 1e-5);

        // At t=0.0, should be 0.0
        let val = signal.at(0.0).unwrap();
        assert!((val[0] - 0.0).abs() < 1e-5);

        // At t=2.0, should be 10.0
        let val = signal.at(2.0).unwrap();
        assert!((val[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_stl_monitor_basic() {
        // φ: x[0] ≥ 5.0
        let phi = STLFormula::greater_eq("x_geq_5", 0, 5.0);
        let monitor = STLMonitor::new(phi, 0.1);

        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![
            Array1::from_vec(vec![6.0]), // robustness = 1.0
            Array1::from_vec(vec![7.0]), // robustness = 2.0
            Array1::from_vec(vec![4.0]), // robustness = -1.0
            Array1::from_vec(vec![8.0]), // robustness = 3.0
        ];

        let signal = Signal::new(times, values);
        let results = monitor.monitor(&signal);

        assert_eq!(results.len(), 4);
        assert_eq!(results[0].1, 1.0);
        assert_eq!(results[1].1, 2.0);
        assert_eq!(results[2].1, -1.0);
        assert_eq!(results[3].1, 3.0);

        assert!(!monitor.satisfies(&signal)); // One point violates
    }

    #[test]
    fn test_stl_eventually() {
        // ◇[0,2] (x[0] ≥ 8.0): eventually x[0] reaches 8.0 within 2 time units
        let phi = STLFormula::greater_eq("x_geq_8", 0, 8.0);
        let eventually_phi = STLFormula::eventually(TimeInterval::new(0.0, 2.0), phi);
        let monitor = STLMonitor::new(eventually_phi, 0.1);

        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![
            Array1::from_vec(vec![5.0]),
            Array1::from_vec(vec![9.0]), // Satisfies at t=1.0
            Array1::from_vec(vec![6.0]),
            Array1::from_vec(vec![7.0]),
        ];

        let signal = Signal::new(times, values);
        let results = monitor.monitor(&signal);

        // At t=0, eventually[0,2] checks t ∈ [0,2], should find 9.0 at t=1
        assert!(results[0].1 >= 0.0);
    }

    #[test]
    fn test_stl_always() {
        // □[0,2] (x[0] ≤ 10.0): always x[0] stays below 10.0 for 2 time units
        let phi = STLFormula::less_eq("x_leq_10", 0, 10.0);
        let always_phi = STLFormula::always(TimeInterval::new(0.0, 2.0), phi);
        let monitor = STLMonitor::new(always_phi, 0.1);

        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![
            Array1::from_vec(vec![8.0]),
            Array1::from_vec(vec![9.0]),
            Array1::from_vec(vec![7.0]),
            Array1::from_vec(vec![15.0]), // Violates at t=3
        ];

        let signal = Signal::new(times, values);

        // At t=0, always[0,2] checks t ∈ [0,2], all values ≤ 10
        let results = monitor.monitor(&signal);
        assert!(results[0].1 >= 0.0);
    }

    #[test]
    fn test_online_monitor() {
        // φ: x[0] ≥ 5.0
        let phi = STLFormula::greater_eq("x_geq_5", 0, 5.0);
        let mut monitor = OnlineSTLMonitor::new(phi);

        // Add observations one by one
        let rob1 = monitor.update(0.0, Array1::from_vec(vec![6.0]));
        assert_eq!(rob1, 1.0);

        let rob2 = monitor.update(1.0, Array1::from_vec(vec![7.0]));
        assert_eq!(rob2, 2.0);

        let rob3 = monitor.update(2.0, Array1::from_vec(vec![4.0]));
        assert_eq!(rob3, -1.0);

        assert!(!monitor.check()); // Current state violates
    }

    #[test]
    fn test_complex_formula() {
        // (x[0] ≥ 5.0) ∧ (x[1] ≤ 10.0)
        let phi1 = STLFormula::greater_eq("x0_geq_5", 0, 5.0);
        let phi2 = STLFormula::less_eq("x1_leq_10", 1, 10.0);
        let phi_complex = STLFormula::and(phi1, phi2);

        let monitor = STLMonitor::new(phi_complex, 0.1);

        let times = vec![0.0, 1.0];
        let values = vec![
            Array1::from_vec(vec![6.0, 9.0]),  // Both satisfied
            Array1::from_vec(vec![7.0, 11.0]), // Second violated
        ];

        let signal = Signal::new(times, values);
        let results = monitor.monitor(&signal);

        assert!(results[0].1 >= 0.0); // t=0: satisfied
        assert!(results[1].1 < 0.0); // t=1: violated
    }

    #[test]
    fn test_horizon_calculation() {
        let phi = STLFormula::greater_eq("x_geq_5", 0, 5.0);
        assert_eq!(phi.horizon(), 0.0);

        let eventually_phi = STLFormula::eventually(TimeInterval::new(0.0, 5.0), phi.clone());
        assert_eq!(eventually_phi.horizon(), 5.0);

        let always_eventually = STLFormula::always(
            TimeInterval::new(0.0, 3.0),
            STLFormula::eventually(TimeInterval::new(0.0, 2.0), phi),
        );
        assert_eq!(always_eventually.horizon(), 5.0); // 3.0 + 2.0
    }
}
