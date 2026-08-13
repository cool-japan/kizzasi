//! Temporal logic constraints for inference
//!
//! Provides temporal logic (LTL/STL) constraint enforcement for time-series predictions.
//! This allows specifying constraints that must hold over time windows.
//!
//! # Temporal Logic Types
//!
//! - **LTL (Linear Temporal Logic)**: Discrete-time temporal properties
//!   - `Always`: φ must hold at all future steps
//!   - `Eventually`: φ must hold at some future step
//!   - `Until`: φ₁ holds until φ₂ becomes true
//!   - `Next`: φ holds at the next step
//!
//! - **STL (Signal Temporal Logic)**: Real-valued continuous-time signals
//!   - Quantitative semantics (robustness)
//!   - Time-bounded operators
//!   - Supports hybrid systems
//!
//! # Examples
//!
//! ## LTL Constraints
//!
//! ```rust,ignore
//! use kizzasi_inference::temporal::{LTLFormula, TemporalConstraint};
//!
//! // "Always x > 0"
//! let always_positive = LTLFormula::Always(
//!     Box::new(LTLFormula::Atomic(|x| x[0] > 0.0))
//! );
//!
//! // "Eventually x > 10"
//! let eventually_large = LTLFormula::Eventually(
//!     Box::new(LTLFormula::Atomic(|x| x[0] > 10.0))
//! );
//! ```
//!
//! ## STL Constraints
//!
//! ```rust,ignore
//! use kizzasi_inference::temporal::{STLFormula, TemporalBound};
//!
//! // "Always[0, 10] (x > 5)"
//! let bounded_constraint = STLFormula::Always {
//!     formula: Box::new(STLFormula::Predicate(|x| x[0] - 5.0)),
//!     bound: TemporalBound::new(0.0, 10.0),
//! };
//! ```

use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Linear Temporal Logic formula
#[derive(Clone)]
pub enum LTLFormula {
    /// Atomic proposition (predicate on signal)
    Atomic(fn(&Array1<f32>) -> bool),

    /// Negation: ¬φ
    Not(Box<LTLFormula>),

    /// Conjunction: φ₁ ∧ φ₂
    And(Box<LTLFormula>, Box<LTLFormula>),

    /// Disjunction: φ₁ ∨ φ₂
    Or(Box<LTLFormula>, Box<LTLFormula>),

    /// Next: ○φ (φ holds at next step)
    Next(Box<LTLFormula>),

    /// Always: □φ (φ holds at all future steps)
    Always(Box<LTLFormula>),

    /// Eventually: ◇φ (φ holds at some future step)
    Eventually(Box<LTLFormula>),

    /// Until: φ₁ U φ₂ (φ₁ holds until φ₂ becomes true)
    Until(Box<LTLFormula>, Box<LTLFormula>),
}

impl LTLFormula {
    /// Check if formula holds on a trace
    pub fn check(&self, trace: &[Array1<f32>], position: usize) -> bool {
        match self {
            LTLFormula::Atomic(pred) => {
                if position < trace.len() {
                    pred(&trace[position])
                } else {
                    false
                }
            }
            LTLFormula::Not(phi) => !phi.check(trace, position),
            LTLFormula::And(phi1, phi2) => {
                phi1.check(trace, position) && phi2.check(trace, position)
            }
            LTLFormula::Or(phi1, phi2) => {
                phi1.check(trace, position) || phi2.check(trace, position)
            }
            LTLFormula::Next(phi) => {
                if position + 1 < trace.len() {
                    phi.check(trace, position + 1)
                } else {
                    false
                }
            }
            LTLFormula::Always(phi) => {
                // Check if φ holds at all positions from current to end
                (position..trace.len()).all(|i| phi.check(trace, i))
            }
            LTLFormula::Eventually(phi) => {
                // Check if φ holds at some position from current to end
                (position..trace.len()).any(|i| phi.check(trace, i))
            }
            LTLFormula::Until(phi1, phi2) => {
                // φ₁ holds until φ₂ becomes true
                for i in position..trace.len() {
                    if phi2.check(trace, i) {
                        return true;
                    }
                    if !phi1.check(trace, i) {
                        return false;
                    }
                }
                false
            }
        }
    }
}

/// Temporal bound for STL formulas
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TemporalBound {
    /// Lower bound (time steps or continuous time)
    pub lower: f32,

    /// Upper bound
    pub upper: f32,
}

impl TemporalBound {
    /// Create a new temporal bound
    pub fn new(lower: f32, upper: f32) -> Self {
        Self { lower, upper }
    }

    /// Check if time is within bound
    pub fn contains(&self, time: f32) -> bool {
        time >= self.lower && time <= self.upper
    }
}

/// Signal Temporal Logic formula
#[derive(Clone)]
pub enum STLFormula {
    /// Predicate: returns robustness value (distance to satisfaction)
    Predicate(fn(&Array1<f32>) -> f32),

    /// Negation: ¬φ
    Not(Box<STLFormula>),

    /// Conjunction: φ₁ ∧ φ₂ (min robustness)
    And(Box<STLFormula>, Box<STLFormula>),

    /// Disjunction: φ₁ ∨ φ₂ (max robustness)
    Or(Box<STLFormula>, Box<STLFormula>),

    /// Always: `□[a,b] φ`
    Always {
        formula: Box<STLFormula>,
        bound: TemporalBound,
    },

    /// Eventually: `◇[a,b] φ`
    Eventually {
        formula: Box<STLFormula>,
        bound: TemporalBound,
    },

    /// Until: `φ₁ U[a,b] φ₂`
    Until {
        phi1: Box<STLFormula>,
        phi2: Box<STLFormula>,
        bound: TemporalBound,
    },
}

impl STLFormula {
    /// Compute robustness (quantitative satisfaction)
    /// Positive = satisfied, negative = violated, magnitude = margin
    pub fn robustness(&self, trace: &[Array1<f32>], time: f32) -> f32 {
        let idx = time as usize;

        match self {
            STLFormula::Predicate(pred) => {
                if idx < trace.len() {
                    pred(&trace[idx])
                } else {
                    f32::NEG_INFINITY
                }
            }
            STLFormula::Not(phi) => -phi.robustness(trace, time),
            STLFormula::And(phi1, phi2) => phi1
                .robustness(trace, time)
                .min(phi2.robustness(trace, time)),
            STLFormula::Or(phi1, phi2) => phi1
                .robustness(trace, time)
                .max(phi2.robustness(trace, time)),
            STLFormula::Always { formula, bound } => {
                // `bound` is inclusive on both ends (`TemporalBound::contains`
                // uses `>=`/`<=`), so the window is `start..=end`, clamped to
                // the last valid trace index rather than excluded at
                // `trace.len()`.
                let start = (time + bound.lower).max(0.0) as usize;
                let end = ((time + bound.upper) as usize).min(trace.len().saturating_sub(1));

                (start..=end)
                    .map(|i| formula.robustness(trace, i as f32))
                    .fold(f32::INFINITY, f32::min)
            }
            STLFormula::Eventually { formula, bound } => {
                let start = (time + bound.lower).max(0.0) as usize;
                let end = ((time + bound.upper) as usize).min(trace.len().saturating_sub(1));

                (start..=end)
                    .map(|i| formula.robustness(trace, i as f32))
                    .fold(f32::NEG_INFINITY, f32::max)
            }
            STLFormula::Until { phi1, phi2, bound } => {
                let start = (time + bound.lower).max(0.0) as usize;
                let end = ((time + bound.upper) as usize).min(trace.len().saturating_sub(1));

                let mut max_rob = f32::NEG_INFINITY;
                for i in start..=end {
                    let rob2 = phi2.robustness(trace, i as f32);
                    // Standard quantitative STL until semantics (Donzé & Maler 2010):
                    // phi1 must hold over the entire prefix [t, t'], not just [t+a, t']
                    let prefix_start = time as usize;
                    let min_rob1 = (prefix_start..i)
                        .map(|j| phi1.robustness(trace, j as f32))
                        .fold(f32::INFINITY, f32::min);
                    max_rob = max_rob.max(rob2.min(min_rob1));
                }
                max_rob
            }
        }
    }

    /// Check if formula is satisfied (robustness >= 0)
    pub fn is_satisfied(&self, trace: &[Array1<f32>], time: f32) -> bool {
        self.robustness(trace, time) >= 0.0
    }

    /// The largest forward time-shift this formula (or any subformula) can
    /// reference, relative to the time it is evaluated at.
    ///
    /// A plain `Predicate` looks only at the exact time it is evaluated at
    /// (shift `0.0`). `Always`/`Eventually` shift by `bound.upper` before
    /// evaluating their subformula, and `Until` by `bound.upper` before
    /// evaluating either subformula; nested bounded operators compound.
    ///
    /// Used by [`TemporalConstraintEnforcer`] to pick an evaluation time
    /// early enough in a live (append-only, sliding-window) trace that a
    /// bounded formula's window is fully populated instead of silently empty
    /// — evaluating a `□[a,b]` formula at the very latest observed sample
    /// leaves no room for the `[a,b]` window to look forward into, which
    /// made every bounded constraint vacuously true.
    pub fn max_upper_bound(&self) -> f32 {
        match self {
            STLFormula::Predicate(_) => 0.0,
            STLFormula::Not(phi) => phi.max_upper_bound(),
            STLFormula::And(phi1, phi2) | STLFormula::Or(phi1, phi2) => {
                phi1.max_upper_bound().max(phi2.max_upper_bound())
            }
            STLFormula::Always { formula, bound } | STLFormula::Eventually { formula, bound } => {
                bound.upper.max(0.0) + formula.max_upper_bound()
            }
            STLFormula::Until { phi1, phi2, bound } => {
                bound.upper.max(0.0) + phi1.max_upper_bound().max(phi2.max_upper_bound())
            }
        }
    }
}

/// Temporal constraint enforcer
pub struct TemporalConstraintEnforcer {
    /// LTL formulas to enforce
    ltl_formulas: Vec<LTLFormula>,

    /// STL formulas to enforce
    stl_formulas: Vec<STLFormula>,

    /// Trace buffer (sliding window)
    trace: VecDeque<Array1<f32>>,

    /// Maximum trace length
    max_trace_len: usize,

    /// Current time step
    current_time: usize,
}

impl TemporalConstraintEnforcer {
    /// Create a new temporal constraint enforcer
    pub fn new(max_trace_len: usize) -> Self {
        Self {
            ltl_formulas: Vec::new(),
            stl_formulas: Vec::new(),
            trace: VecDeque::new(),
            max_trace_len,
            current_time: 0,
        }
    }

    /// Add an LTL formula
    pub fn add_ltl(&mut self, formula: LTLFormula) {
        self.ltl_formulas.push(formula);
    }

    /// Add an STL formula
    pub fn add_stl(&mut self, formula: STLFormula) {
        self.stl_formulas.push(formula);
    }

    /// Update trace with new observation
    pub fn update(&mut self, signal: Array1<f32>) {
        self.trace.push_back(signal);
        if self.trace.len() > self.max_trace_len {
            self.trace.pop_front();
        }
        self.current_time += 1;
    }

    /// Check if all LTL constraints are satisfied
    pub fn check_ltl(&self) -> bool {
        if self.trace.is_empty() {
            return true;
        }

        let trace_vec: Vec<_> = self.trace.iter().cloned().collect();
        self.ltl_formulas
            .iter()
            .all(|formula| formula.check(&trace_vec, 0))
    }

    /// Check if all STL constraints are satisfied
    ///
    /// Each formula is evaluated at `latest - formula.max_upper_bound()`
    /// (clamped to `0.0`), not uniformly at the latest sample: a bounded
    /// formula (e.g. `□[0,2] φ`) needs `bound.upper` samples *beyond* its
    /// evaluation point to have a non-vacuous window, which a live,
    /// append-only trace can only supply by evaluating that many samples
    /// back from the most recent one. A plain, unbounded `Predicate` has
    /// `max_upper_bound() == 0.0` and is still evaluated at the latest
    /// sample, exactly as before.
    pub fn check_stl(&self) -> bool {
        if self.trace.is_empty() {
            return true;
        }

        let trace_vec: Vec<_> = self.trace.iter().cloned().collect();
        let latest_time = (trace_vec.len() - 1) as f32;
        self.stl_formulas.iter().all(|formula| {
            let eval_time = (latest_time - formula.max_upper_bound()).max(0.0);
            formula.is_satisfied(&trace_vec, eval_time)
        })
    }

    /// Get STL robustness values
    ///
    /// See [`TemporalConstraintEnforcer::check_stl`]: each formula is
    /// evaluated at its own bound-aware time, not uniformly at the latest
    /// sample.
    pub fn stl_robustness(&self) -> Vec<f32> {
        if self.trace.is_empty() {
            return vec![];
        }

        let trace_vec: Vec<_> = self.trace.iter().cloned().collect();
        let latest_time = (trace_vec.len() - 1) as f32;
        self.stl_formulas
            .iter()
            .map(|formula| {
                let eval_time = (latest_time - formula.max_upper_bound()).max(0.0);
                formula.robustness(&trace_vec, eval_time)
            })
            .collect()
    }

    /// Check if all temporal constraints are satisfied
    pub fn check_all(&self) -> bool {
        self.check_ltl() && self.check_stl()
    }

    /// Reset the enforcer
    pub fn reset(&mut self) {
        self.trace.clear();
        self.current_time = 0;
    }

    /// Get current trace length
    pub fn trace_length(&self) -> usize {
        self.trace.len()
    }

    /// Total number of observations fed via
    /// [`TemporalConstraintEnforcer::update`] since creation or the last
    /// [`TemporalConstraintEnforcer::reset`].
    ///
    /// Unlike [`TemporalConstraintEnforcer::trace_length`], this does not
    /// shrink when the sliding window (`max_trace_len`) evicts old samples —
    /// it is a monotonic step counter.
    pub fn current_time(&self) -> usize {
        self.current_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ltl_atomic() {
        let positive = LTLFormula::Atomic(|x| x[0] > 0.0);

        let trace = vec![
            Array1::from_vec(vec![1.0]),
            Array1::from_vec(vec![2.0]),
            Array1::from_vec(vec![3.0]),
        ];

        assert!(positive.check(&trace, 0));
        assert!(positive.check(&trace, 1));
        assert!(positive.check(&trace, 2));

        let trace_neg = vec![
            Array1::from_vec(vec![1.0]),
            Array1::from_vec(vec![-1.0]),
            Array1::from_vec(vec![2.0]),
        ];

        assert!(!positive.check(&trace_neg, 1));
    }

    #[test]
    fn test_ltl_always() {
        let always_positive = LTLFormula::Always(Box::new(LTLFormula::Atomic(|x| x[0] > 0.0)));

        let trace_good = vec![
            Array1::from_vec(vec![1.0]),
            Array1::from_vec(vec![2.0]),
            Array1::from_vec(vec![3.0]),
        ];

        assert!(always_positive.check(&trace_good, 0));

        let trace_bad = vec![
            Array1::from_vec(vec![1.0]),
            Array1::from_vec(vec![-1.0]),
            Array1::from_vec(vec![2.0]),
        ];

        assert!(!always_positive.check(&trace_bad, 0));
    }

    #[test]
    fn test_ltl_eventually() {
        let eventually_large =
            LTLFormula::Eventually(Box::new(LTLFormula::Atomic(|x| x[0] > 10.0)));

        let trace = vec![
            Array1::from_vec(vec![1.0]),
            Array1::from_vec(vec![5.0]),
            Array1::from_vec(vec![15.0]),
        ];

        assert!(eventually_large.check(&trace, 0));

        let trace_bad = vec![
            Array1::from_vec(vec![1.0]),
            Array1::from_vec(vec![5.0]),
            Array1::from_vec(vec![9.0]),
        ];

        assert!(!eventually_large.check(&trace_bad, 0));
    }

    #[test]
    fn test_stl_predicate() {
        let pred = STLFormula::Predicate(|x| x[0] - 5.0);

        let trace = vec![Array1::from_vec(vec![10.0]), Array1::from_vec(vec![3.0])];

        // 10 - 5 = 5 (satisfied)
        assert_eq!(pred.robustness(&trace, 0.0), 5.0);

        // 3 - 5 = -2 (violated)
        assert_eq!(pred.robustness(&trace, 1.0), -2.0);

        assert!(pred.is_satisfied(&trace, 0.0));
        assert!(!pred.is_satisfied(&trace, 1.0));
    }

    #[test]
    fn test_stl_always() {
        let always_large = STLFormula::Always {
            formula: Box::new(STLFormula::Predicate(|x| x[0] - 5.0)),
            bound: TemporalBound::new(0.0, 3.0),
        };

        let trace_good = vec![
            Array1::from_vec(vec![10.0]),
            Array1::from_vec(vec![8.0]),
            Array1::from_vec(vec![7.0]),
        ];

        // Min robustness = 7 - 5 = 2
        assert_eq!(always_large.robustness(&trace_good, 0.0), 2.0);

        let trace_bad = vec![
            Array1::from_vec(vec![10.0]),
            Array1::from_vec(vec![3.0]),
            Array1::from_vec(vec![8.0]),
        ];

        // Min robustness = 3 - 5 = -2
        assert_eq!(always_large.robustness(&trace_bad, 0.0), -2.0);
    }

    #[test]
    fn test_temporal_bound() {
        let bound = TemporalBound::new(2.0, 5.0);

        assert!(!bound.contains(1.0));
        assert!(bound.contains(2.0));
        assert!(bound.contains(3.5));
        assert!(bound.contains(5.0));
        assert!(!bound.contains(6.0));
    }

    #[test]
    fn test_temporal_enforcer() {
        let mut enforcer = TemporalConstraintEnforcer::new(10);

        // Add LTL: always positive
        let always_positive = LTLFormula::Always(Box::new(LTLFormula::Atomic(|x| x[0] > 0.0)));
        enforcer.add_ltl(always_positive);

        // Update with positive values
        enforcer.update(Array1::from_vec(vec![1.0]));
        enforcer.update(Array1::from_vec(vec![2.0]));
        enforcer.update(Array1::from_vec(vec![3.0]));

        assert!(enforcer.check_ltl());
        assert_eq!(enforcer.trace_length(), 3);

        // Add negative value
        enforcer.update(Array1::from_vec(vec![-1.0]));
        assert!(!enforcer.check_ltl());

        // Reset
        enforcer.reset();
        assert_eq!(enforcer.trace_length(), 0);
    }

    #[test]
    fn test_stl_enforcer() {
        let mut enforcer = TemporalConstraintEnforcer::new(10);

        // Add STL: x > 5
        let predicate = STLFormula::Predicate(|x| x[0] - 5.0);
        enforcer.add_stl(predicate);

        enforcer.update(Array1::from_vec(vec![10.0]));
        assert!(enforcer.check_stl());

        enforcer.update(Array1::from_vec(vec![3.0]));
        assert!(!enforcer.check_stl());

        let robustness = enforcer.stl_robustness();
        assert_eq!(robustness.len(), 1);
    }

    #[test]
    fn test_stl_until_basic_satisfied() {
        // phi1: x[0] >= 0 (robustness = x[0], non-negative for x >= 0)
        // phi2: x[0] >= 2 (robustness = x[0] - 2.0, positive once x >= 2)
        // trace: 1, 2, 3, 4, 5 — phi1 holds (all positive), phi2 becomes true at t=1 (x=2)
        // Using strictly positive values so that min over prefix is > 0
        let trace: Vec<Array1<f32>> = (1..=5).map(|i| Array1::from_vec(vec![i as f32])).collect();

        let phi1 = STLFormula::Predicate(|x| x[0] - 0.5); // positive for all values >= 0.5
        let phi2 = STLFormula::Predicate(|x| x[0] - 2.0);
        let until = STLFormula::Until {
            phi1: Box::new(phi1),
            phi2: Box::new(phi2),
            bound: TemporalBound::new(0.0, 5.0),
        };
        // At i=1 (x=2): rob2=0.0, prefix [0..1): phi1(1)=0.5 → min=0.5 → combined=0.0
        // At i=2 (x=3): rob2=1.0, prefix [0..2): phi1(1)=0.5,phi1(2)=1.5 → min=0.5 → combined=0.5
        // max_rob = 0.5 > 0
        let rob = until.robustness(&trace, 0.0);
        assert!(
            rob > 0.0,
            "Until should be satisfied with positive margin, robustness={}",
            rob
        );
        assert!(until.is_satisfied(&trace, 0.0), "Until should be satisfied");
    }

    #[test]
    fn test_stl_until_prefix_violation_detected() {
        // phi1 FAILS at t=0 (the prefix), phi2 holds at t=2.
        // With bound.lower=1: the buggy code would miss the t=0 failure.
        // With the fix (start from t=0), robustness is reduced/negative.
        //
        // trace: t=0: -1.0 (phi1 fails: x < 0), t=1: 1.0 (phi1 holds), t=2..4: 5.0 (phi2 holds)
        let trace: Vec<Array1<f32>> = vec![
            Array1::from_vec(vec![-1.0_f32]), // t=0: phi1 fails (x - 0 = -1 < 0)
            Array1::from_vec(vec![1.0_f32]),  // t=1: phi1 holds
            Array1::from_vec(vec![5.0_f32]),  // t=2: phi2 holds (x - 5 = 0)
            Array1::from_vec(vec![5.0_f32]),  // t=3
            Array1::from_vec(vec![5.0_f32]),  // t=4
        ];
        let phi1 = STLFormula::Predicate(|x| x[0]);
        let phi2 = STLFormula::Predicate(|x| x[0] - 5.0);
        // bound.lower=1: window starts at t=1, but phi1 must hold from t=0
        let until = STLFormula::Until {
            phi1: Box::new(phi1),
            phi2: Box::new(phi2),
            bound: TemporalBound::new(1.0, 4.0),
        };
        let rob = until.robustness(&trace, 0.0);
        // With the fix: phi1 infimum over [0, t'] includes t=0 → robustness should be <= 0.0
        assert!(
            rob <= 0.0,
            "Prefix violation should reduce robustness, got rob={}",
            rob
        );
    }

    #[test]
    fn test_stl_until_phi2_never_holds() {
        // phi2 never becomes true → Until is not satisfied → negative robustness
        let trace: Vec<Array1<f32>> = (0..5).map(|_| Array1::from_vec(vec![1.0_f32])).collect();
        let phi1 = STLFormula::Predicate(|x| x[0]);
        let phi2 = STLFormula::Predicate(|x| x[0] - 10.0); // never reached (values are 1.0)
        let until = STLFormula::Until {
            phi1: Box::new(phi1),
            phi2: Box::new(phi2),
            bound: TemporalBound::new(0.0, 4.0),
        };
        let rob = until.robustness(&trace, 0.0);
        assert!(
            rob < 0.0,
            "phi2 never holds → should be unsatisfied, got rob={}",
            rob
        );
    }

    /// Regression: bounded operators used an exclusive upper end
    /// (`start..end`), so a violating sample at exactly index `bound.upper`
    /// was silently excluded from the checked window.
    #[test]
    fn test_stl_always_bounded_window_includes_upper_index() {
        let bound = TemporalBound::new(0.0, 2.0);
        let always = STLFormula::Always {
            formula: Box::new(STLFormula::Predicate(|x| x[0])), // satisfied when x >= 0
            bound,
        };

        // Indices 0,1 are non-negative; index 2 (== bound.upper) is negative.
        let trace = vec![
            Array1::from_vec(vec![1.0_f32]),
            Array1::from_vec(vec![1.0_f32]),
            Array1::from_vec(vec![-1.0_f32]),
            Array1::from_vec(vec![1.0_f32]),
            Array1::from_vec(vec![1.0_f32]),
        ];

        assert!(
            !always.is_satisfied(&trace, 0.0),
            "a violation at exactly index `upper` must be caught by the bounded window"
        );
        assert!(
            always.robustness(&trace, 0.0) < 0.0,
            "robustness must reflect the violation at index upper, got {}",
            always.robustness(&trace, 0.0)
        );
    }

    /// Regression: `check_stl`/`stl_robustness` used to evaluate every
    /// formula at the trace's latest sample, so any bounded window
    /// (`bound.upper > 0`) looked forward past the end of the trace and
    /// folded over an empty range — vacuously satisfied (`Always`) no matter
    /// what the trace contained. A bounded `Always` violated within the
    /// available trace must now actually be detected.
    #[test]
    fn test_check_stl_detects_bounded_violation_not_vacuous() {
        let mut enforcer = TemporalConstraintEnforcer::new(10);
        enforcer.add_stl(STLFormula::Always {
            formula: Box::new(STLFormula::Predicate(|x| x[0])),
            bound: TemporalBound::new(0.0, 2.0),
        });

        for v in [1.0_f32, 1.0, -1.0, 1.0, 1.0] {
            enforcer.update(Array1::from_vec(vec![v]));
        }

        assert!(
            !enforcer.check_stl(),
            "a bounded Always window containing a violation must not be vacuously satisfied"
        );
        let robustness = enforcer.stl_robustness();
        assert_eq!(robustness.len(), 1);
        assert!(
            robustness[0] < 0.0,
            "robustness must reflect the violation, got {}",
            robustness[0]
        );
    }

    /// An unbounded `Predicate` (the common case) must still be evaluated at
    /// the most recent sample, exactly as before — `max_upper_bound() == 0`
    /// means the bound-aware eval time collapses to the latest index.
    #[test]
    fn test_check_stl_unbounded_predicate_still_uses_latest_sample() {
        let mut enforcer = TemporalConstraintEnforcer::new(10);
        enforcer.add_stl(STLFormula::Predicate(|x| x[0] - 5.0));

        enforcer.update(Array1::from_vec(vec![10.0]));
        assert!(enforcer.check_stl());

        enforcer.update(Array1::from_vec(vec![3.0]));
        assert!(!enforcer.check_stl());
    }

    /// Regression: `current_time` was incremented and reset but never read
    /// anywhere. It must now be a real, monotonic step counter distinct from
    /// `trace_length` (which shrinks once the sliding window evicts old
    /// samples).
    #[test]
    fn test_current_time_tracks_total_updates_beyond_window() {
        let mut enforcer = TemporalConstraintEnforcer::new(3);
        for i in 0..7 {
            enforcer.update(Array1::from_vec(vec![i as f32]));
        }
        assert_eq!(enforcer.current_time(), 7);
        assert_eq!(enforcer.trace_length(), 3); // capped by max_trace_len

        enforcer.reset();
        assert_eq!(enforcer.current_time(), 0);
    }

    #[test]
    fn test_stl_until_is_satisfied_agrees_with_robustness() {
        // is_satisfied must agree with robustness >= 0
        let trace: Vec<Array1<f32>> = (0..5)
            .map(|i| Array1::from_vec(vec![i as f32 + 1.0]))
            .collect();
        let phi1 = STLFormula::Predicate(|x| x[0]);
        let phi2 = STLFormula::Predicate(|x| x[0] - 3.0);
        let until = STLFormula::Until {
            phi1: Box::new(phi1),
            phi2: Box::new(phi2),
            bound: TemporalBound::new(0.0, 4.0),
        };
        let rob = until.robustness(&trace, 0.0);
        let sat = until.is_satisfied(&trace, 0.0);
        assert_eq!(
            sat,
            rob >= 0.0,
            "is_satisfied disagrees with robustness={}",
            rob
        );
    }
}
