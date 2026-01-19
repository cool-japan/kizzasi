use serde::{Deserialize, Serialize};

use super::Constraint;

// ============================================================================
// Linear Temporal Logic (LTL) Constraints
// ============================================================================

/// Linear Temporal Logic operators for expressing temporal properties
///
/// LTL allows specification of properties that must hold over time traces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LTLOperator {
    /// Always (G): property φ must hold at all future time steps
    /// G(φ) = φ ∧ X(G(φ))
    Always(Box<LTLFormula>),

    /// Eventually (F): property φ must hold at some future time step
    /// F(φ) = φ ∨ X(F(φ))
    Eventually(Box<LTLFormula>),

    /// Next (X): property φ must hold at the next time step
    Next(Box<LTLFormula>),

    /// Until (U): φ holds until ψ becomes true
    /// φ U ψ = ψ ∨ (φ ∧ X(φ U ψ))
    Until(Box<LTLFormula>, Box<LTLFormula>),

    /// Release (R): ψ holds until and including when φ becomes true
    /// φ R ψ = ψ ∧ (φ ∨ X(φ R ψ))
    Release(Box<LTLFormula>, Box<LTLFormula>),

    /// And: both properties must hold
    And(Box<LTLFormula>, Box<LTLFormula>),

    /// Or: at least one property must hold
    Or(Box<LTLFormula>, Box<LTLFormula>),

    /// Not: property must not hold
    Not(Box<LTLFormula>),
}

/// LTL formula combining operators and atomic propositions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LTLFormula {
    /// Atomic proposition: a constraint that must be satisfied
    Atom(Constraint),

    /// Temporal operator
    Operator(LTLOperator),
}

impl LTLFormula {
    /// Create an Always formula: G(φ)
    pub fn always(formula: LTLFormula) -> Self {
        Self::Operator(LTLOperator::Always(Box::new(formula)))
    }

    /// Create an Eventually formula: F(φ)
    pub fn eventually(formula: LTLFormula) -> Self {
        Self::Operator(LTLOperator::Eventually(Box::new(formula)))
    }

    /// Create a Next formula: X(φ)
    pub fn next(formula: LTLFormula) -> Self {
        Self::Operator(LTLOperator::Next(Box::new(formula)))
    }

    /// Create an Until formula: φ U ψ
    pub fn until(phi: LTLFormula, psi: LTLFormula) -> Self {
        Self::Operator(LTLOperator::Until(Box::new(phi), Box::new(psi)))
    }

    /// Create a Release formula: φ R ψ
    pub fn release(phi: LTLFormula, psi: LTLFormula) -> Self {
        Self::Operator(LTLOperator::Release(Box::new(phi), Box::new(psi)))
    }

    /// Create an And formula
    pub fn and(left: LTLFormula, right: LTLFormula) -> Self {
        Self::Operator(LTLOperator::And(Box::new(left), Box::new(right)))
    }

    /// Create an Or formula
    pub fn or(left: LTLFormula, right: LTLFormula) -> Self {
        Self::Operator(LTLOperator::Or(Box::new(left), Box::new(right)))
    }

    /// Create a Not formula (negation)
    #[allow(clippy::should_implement_trait)]
    pub fn not(formula: LTLFormula) -> Self {
        Self::Operator(LTLOperator::Not(Box::new(formula)))
    }

    /// Create an atomic formula from a constraint
    pub fn atom(constraint: Constraint) -> Self {
        Self::Atom(constraint)
    }
}

/// LTL constraint checker that evaluates formulas over time traces
#[derive(Debug, Clone)]
pub struct LTLChecker {
    formula: LTLFormula,
    trace: Vec<f32>,
    max_trace_length: usize,
}

impl LTLChecker {
    /// Create a new LTL checker
    pub fn new(formula: LTLFormula) -> Self {
        Self {
            formula,
            trace: Vec::new(),
            max_trace_length: 100,
        }
    }

    /// Set maximum trace length (for bounded checking)
    pub fn with_max_trace_length(mut self, length: usize) -> Self {
        self.max_trace_length = length;
        self
    }

    /// Push a new observation to the trace
    pub fn push(&mut self, value: f32) {
        self.trace.push(value);
        if self.trace.len() > self.max_trace_length {
            self.trace.remove(0);
        }
    }

    /// Check if the formula is satisfied over the current trace (bounded semantics)
    pub fn check(&self) -> bool {
        if self.trace.is_empty() {
            return true;
        }
        self.check_formula(&self.formula, 0)
    }

    /// Check formula at specific position in trace
    fn check_formula(&self, formula: &LTLFormula, pos: usize) -> bool {
        if pos >= self.trace.len() {
            // Beyond trace: assume satisfied (finite trace semantics)
            return true;
        }

        match formula {
            LTLFormula::Atom(constraint) => constraint.check(self.trace[pos]),
            LTLFormula::Operator(op) => self.check_operator(op, pos),
        }
    }

    /// Check temporal operator at specific position
    fn check_operator(&self, op: &LTLOperator, pos: usize) -> bool {
        match op {
            LTLOperator::Always(phi) => {
                // G(φ): φ must hold at all positions from pos onwards
                (pos..self.trace.len()).all(|i| self.check_formula(phi, i))
            }
            LTLOperator::Eventually(phi) => {
                // F(φ): φ must hold at some position from pos onwards
                (pos..self.trace.len()).any(|i| self.check_formula(phi, i))
            }
            LTLOperator::Next(phi) => {
                // X(φ): φ holds at pos+1
                self.check_formula(phi, pos + 1)
            }
            LTLOperator::Until(phi, psi) => {
                // φ U ψ: φ holds until ψ becomes true
                for i in pos..self.trace.len() {
                    if self.check_formula(psi, i) {
                        // ψ holds at i, check φ held at all j < i
                        return (pos..i).all(|j| self.check_formula(phi, j));
                    }
                }
                // ψ never holds: formula is not satisfied
                false
            }
            LTLOperator::Release(phi, psi) => {
                // φ R ψ: ψ holds until and including when φ becomes true
                for i in pos..self.trace.len() {
                    if self.check_formula(phi, i) {
                        // φ holds at i, check ψ held at all j <= i
                        return (pos..=i).all(|j| self.check_formula(psi, j));
                    }
                    if !self.check_formula(psi, i) {
                        return false;
                    }
                }
                // φ never holds, ψ must hold everywhere
                (pos..self.trace.len()).all(|i| self.check_formula(psi, i))
            }
            LTLOperator::And(left, right) => {
                self.check_formula(left, pos) && self.check_formula(right, pos)
            }
            LTLOperator::Or(left, right) => {
                self.check_formula(left, pos) || self.check_formula(right, pos)
            }
            LTLOperator::Not(phi) => !self.check_formula(phi, pos),
        }
    }

    /// Compute violation score (0 if satisfied, higher for more violations)
    pub fn violation(&self) -> f32 {
        self.count_violations(&self.formula, 0) as f32
    }

    /// Count violations in the trace
    fn count_violations(&self, formula: &LTLFormula, pos: usize) -> usize {
        if pos >= self.trace.len() {
            return 0;
        }

        match formula {
            LTLFormula::Atom(constraint) => {
                if constraint.check(self.trace[pos]) {
                    0
                } else {
                    1
                }
            }
            LTLFormula::Operator(op) => match op {
                LTLOperator::Always(phi) => (pos..self.trace.len())
                    .map(|i| self.count_violations(phi, i))
                    .sum(),
                LTLOperator::Eventually(phi) => {
                    // If any position satisfies, no violation
                    if (pos..self.trace.len()).any(|i| self.check_formula(phi, i)) {
                        0
                    } else {
                        1
                    }
                }
                LTLOperator::Next(phi) => self.count_violations(phi, pos + 1),
                LTLOperator::And(left, right) => {
                    self.count_violations(left, pos) + self.count_violations(right, pos)
                }
                LTLOperator::Or(left, right) => self
                    .count_violations(left, pos)
                    .min(self.count_violations(right, pos)),
                LTLOperator::Not(phi) => {
                    if self.check_formula(phi, pos) {
                        1
                    } else {
                        0
                    }
                }
                _ => {
                    if self.check_operator(op, pos) {
                        0
                    } else {
                        1
                    }
                }
            },
        }
    }

    /// Reset the trace
    pub fn reset(&mut self) {
        self.trace.clear();
    }

    /// Get current trace length
    pub fn trace_len(&self) -> usize {
        self.trace.len()
    }
}
