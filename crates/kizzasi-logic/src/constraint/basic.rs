//! Constraint definitions and builders
//!
//! Provides constraint types for signal bounds and composition operators
//! for building complex constraint expressions, including temporal constraints
//! for rate-of-change limits.

use crate::error::{LogicError, LogicResult};
use serde::{Deserialize, Serialize};

/// Logical operators for combining constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogicalOperator {
    And,
    Or,
    Not,
    Implies,
}

/// Composed constraint from multiple constraints with logical operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComposedConstraint {
    /// Single constraint
    Single(Constraint),
    /// AND of two constraints (both must be satisfied)
    And(Box<ComposedConstraint>, Box<ComposedConstraint>),
    /// OR of two constraints (at least one must be satisfied)
    Or(Box<ComposedConstraint>, Box<ComposedConstraint>),
    /// NOT constraint (must not be satisfied)
    Not(Box<ComposedConstraint>),
    /// Implies: if A then B (equivalent to !A OR B)
    Implies(Box<ComposedConstraint>, Box<ComposedConstraint>),
}

impl ComposedConstraint {
    /// Create a single constraint
    pub fn single(constraint: Constraint) -> Self {
        Self::Single(constraint)
    }

    /// Combine with AND operator
    pub fn and(self, other: ComposedConstraint) -> Self {
        Self::And(Box::new(self), Box::new(other))
    }

    /// Combine with OR operator
    pub fn or(self, other: ComposedConstraint) -> Self {
        Self::Or(Box::new(self), Box::new(other))
    }

    /// Negate this constraint
    pub fn negate(self) -> Self {
        Self::Not(Box::new(self))
    }

    /// Create implication: if self then other
    pub fn implies(self, other: ComposedConstraint) -> Self {
        Self::Implies(Box::new(self), Box::new(other))
    }

    /// Check if a value satisfies this composed constraint
    pub fn check(&self, value: f32) -> bool {
        match self {
            Self::Single(c) => c.check(value),
            Self::And(a, b) => a.check(value) && b.check(value),
            Self::Or(a, b) => a.check(value) || b.check(value),
            Self::Not(c) => !c.check(value),
            Self::Implies(a, b) => !a.check(value) || b.check(value),
        }
    }

    /// Check all dimensions of a multi-dimensional value
    pub fn check_all(&self, values: &[f32]) -> bool {
        match self {
            Self::Single(c) => {
                if let Some(dim) = c.dimension() {
                    values.get(dim).is_some_and(|&v| c.check(v))
                } else {
                    values.iter().all(|&v| c.check(v))
                }
            }
            Self::And(a, b) => a.check_all(values) && b.check_all(values),
            Self::Or(a, b) => a.check_all(values) || b.check_all(values),
            Self::Not(c) => !c.check_all(values),
            Self::Implies(a, b) => !a.check_all(values) || b.check_all(values),
        }
    }

    /// Compute total violation (used for loss computation)
    pub fn violation(&self, value: f32) -> f32 {
        match self {
            Self::Single(c) => c.violation(value),
            Self::And(a, b) => a.violation(value) + b.violation(value),
            Self::Or(a, b) => a.violation(value).min(b.violation(value)),
            Self::Not(c) => {
                // If the constraint is satisfied, it's a violation (inverted)
                if c.check(value) {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Implies(a, b) => {
                // If A is true and B is false, it's a violation
                if a.check(value) && !b.check(value) {
                    b.violation(value)
                } else {
                    0.0
                }
            }
        }
    }

    /// Project a value to satisfy this constraint (best effort)
    pub fn project(&self, value: f32) -> f32 {
        match self {
            Self::Single(c) => c.project(value),
            Self::And(a, b) => {
                // Apply both projections sequentially
                let v1 = a.project(value);
                b.project(v1)
            }
            Self::Or(a, b) => {
                // Choose the projection with smaller change
                let proj_a = a.project(value);
                let proj_b = b.project(value);
                let dist_a = (value - proj_a).abs();
                let dist_b = (value - proj_b).abs();
                if dist_a <= dist_b {
                    proj_a
                } else {
                    proj_b
                }
            }
            Self::Not(_) => {
                // Cannot project onto "not satisfied" region in general
                // Return value unchanged
                value
            }
            Self::Implies(a, b) => {
                // If A is satisfied, must also satisfy B
                if a.check(value) {
                    b.project(value)
                } else {
                    value
                }
            }
        }
    }
}

/// Type of bound constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundType {
    LessThan(f32),
    LessEq(f32),
    GreaterThan(f32),
    GreaterEq(f32),
    Equal(f32, f32), // value, tolerance
    InRange(f32, f32),
}

/// A numeric interval derived from a single [`BoundType`]: the (possibly
/// open, possibly one-sided) endpoints implied by that bound, ignoring any
/// `dimension` tag.
///
/// Used for exact (non-sampling) feasibility and dependency analysis in
/// [`crate::constraint_analysis`], and to validate a bound at construction
/// time in [`ConstraintBuilder::build`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BoundInterval {
    pub(crate) lo: f32,
    pub(crate) lo_inclusive: bool,
    pub(crate) hi: f32,
    pub(crate) hi_inclusive: bool,
}

impl BoundInterval {
    pub(crate) const UNBOUNDED: Self = Self {
        lo: f32::NEG_INFINITY,
        lo_inclusive: false,
        hi: f32::INFINITY,
        hi_inclusive: false,
    };

    /// True when no real number satisfies both endpoints.
    pub(crate) fn is_empty(&self) -> bool {
        if self.lo > self.hi {
            return true;
        }
        if self.lo == self.hi {
            return !(self.lo_inclusive && self.hi_inclusive);
        }
        false
    }

    /// Intersection of two intervals: the tighter bound wins on each side;
    /// on a tie, the side is inclusive only if both operands are.
    pub(crate) fn intersect(&self, other: &Self) -> Self {
        let (lo, lo_inclusive) = match self.lo.partial_cmp(&other.lo) {
            Some(std::cmp::Ordering::Greater) => (self.lo, self.lo_inclusive),
            Some(std::cmp::Ordering::Less) => (other.lo, other.lo_inclusive),
            _ => (self.lo, self.lo_inclusive && other.lo_inclusive),
        };
        let (hi, hi_inclusive) = match self.hi.partial_cmp(&other.hi) {
            Some(std::cmp::Ordering::Less) => (self.hi, self.hi_inclusive),
            Some(std::cmp::Ordering::Greater) => (other.hi, other.hi_inclusive),
            _ => (self.hi, self.hi_inclusive && other.hi_inclusive),
        };
        Self {
            lo,
            lo_inclusive,
            hi,
            hi_inclusive,
        }
    }

    /// A representative point comfortably inside the interval, or `None`
    /// when the interval is empty. Uses a magnitude-scaled step off an
    /// open/one-sided endpoint instead of a fixed offset, avoiding the
    /// "epsilon rounds back to the bound itself" failure this module's
    /// `project` had (see `Constraint::project`).
    pub(crate) fn witness(&self) -> Option<f32> {
        if self.is_empty() {
            return None;
        }
        Some(match (self.lo.is_finite(), self.hi.is_finite()) {
            (true, true) => {
                if self.lo >= self.hi {
                    self.lo
                } else {
                    0.5 * (self.lo + self.hi)
                }
            }
            (true, false) => self.lo + (self.lo.abs() * 1e-3).max(1.0),
            (false, true) => self.hi - (self.hi.abs() * 1e-3).max(1.0),
            (false, false) => 0.0,
        })
    }
}

impl BoundType {
    /// The feasible interval implied by this bound (see [`BoundInterval`]).
    pub(crate) fn interval(&self) -> BoundInterval {
        match self {
            BoundType::LessThan(b) => BoundInterval {
                hi: *b,
                hi_inclusive: false,
                ..BoundInterval::UNBOUNDED
            },
            BoundType::LessEq(b) => BoundInterval {
                hi: *b,
                hi_inclusive: true,
                ..BoundInterval::UNBOUNDED
            },
            BoundType::GreaterThan(b) => BoundInterval {
                lo: *b,
                lo_inclusive: false,
                ..BoundInterval::UNBOUNDED
            },
            BoundType::GreaterEq(b) => BoundInterval {
                lo: *b,
                lo_inclusive: true,
                ..BoundInterval::UNBOUNDED
            },
            BoundType::Equal(target, tol) => BoundInterval {
                lo: target - tol,
                lo_inclusive: true,
                hi: target + tol,
                hi_inclusive: true,
            },
            BoundType::InRange(lo, hi) => BoundInterval {
                lo: *lo,
                lo_inclusive: true,
                hi: *hi,
                hi_inclusive: true,
            },
        }
    }
}

/// Validate that `bound` is well-formed: every numeric field is finite, any
/// tolerance is non-negative, and the implied interval is non-empty.
///
/// Without this check e.g. `ConstraintBuilder::new().name("x").in_range(5.0,
/// 1.0).build()` used to succeed and only fail later — and not with a
/// recoverable error, but a panic inside `f32::clamp` the first time
/// `Constraint::project` ran (`clamp` asserts `min <= max`). A NaN bound is
/// even quieter: every `check`/`violation` call silently disagrees with a
/// human's expectation without ever erroring.
fn validate_bound(bound: &BoundType) -> LogicResult<()> {
    let all_finite = match bound {
        BoundType::LessThan(b)
        | BoundType::LessEq(b)
        | BoundType::GreaterThan(b)
        | BoundType::GreaterEq(b) => b.is_finite(),
        BoundType::Equal(target, tol) => target.is_finite() && tol.is_finite() && *tol >= 0.0,
        BoundType::InRange(lo, hi) => lo.is_finite() && hi.is_finite(),
    };
    if !all_finite {
        return Err(LogicError::InvalidConstraint(format!(
            "bound has a non-finite value or a negative tolerance: {bound:?}"
        )));
    }
    if bound.interval().is_empty() {
        return Err(LogicError::InvalidConstraint(format!(
            "bound describes an empty feasible region: {bound:?}"
        )));
    }
    Ok(())
}

/// A single constraint on signal values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    name: String,
    dimension: Option<usize>,
    bound: BoundType,
    weight: f32,
}

impl Constraint {
    /// Check if a value satisfies this constraint
    pub fn check(&self, value: f32) -> bool {
        match &self.bound {
            BoundType::LessThan(b) => value < *b,
            BoundType::LessEq(b) => value <= *b,
            BoundType::GreaterThan(b) => value > *b,
            BoundType::GreaterEq(b) => value >= *b,
            BoundType::Equal(target, tol) => (value - target).abs() <= *tol,
            BoundType::InRange(lo, hi) => value >= *lo && value <= *hi,
        }
    }

    /// Compute the violation amount (0 if satisfied)
    pub fn violation(&self, value: f32) -> f32 {
        match &self.bound {
            BoundType::LessThan(b) | BoundType::LessEq(b) => (value - b).max(0.0),
            BoundType::GreaterThan(b) | BoundType::GreaterEq(b) => (b - value).max(0.0),
            BoundType::Equal(target, _) => (value - target).abs(),
            BoundType::InRange(lo, hi) => {
                if value < *lo {
                    lo - value
                } else if value > *hi {
                    value - hi
                } else {
                    0.0
                }
            }
        }
    }

    /// Project a value onto the valid region
    ///
    /// For strict bounds (`LessThan`/`GreaterThan`) the projected value is
    /// nudged by exactly one representable `f32` step using
    /// [`f32::next_down`]/[`f32::next_up`], not a fixed `f32::EPSILON`
    /// offset. `f32::EPSILON` is the gap between `1.0` and the next
    /// representable float; once a bound's own ULP exceeds that (any
    /// `|b| >= 2.0`), `b - f32::EPSILON` rounds back to exactly `b`, so the
    /// projected point would still fail `Constraint::check`. `next_down`/
    /// `next_up` compute the true adjacent representable value regardless of
    /// magnitude, and pass NaN/±∞ through unchanged rather than panicking,
    /// so a bound built from untrusted data (`BoundType` derives
    /// `Deserialize` and so bypasses `ConstraintBuilder`'s validation) can
    /// never make this function panic.
    pub fn project(&self, value: f32) -> f32 {
        match &self.bound {
            BoundType::LessThan(b) => value.min(b.next_down()),
            BoundType::LessEq(b) => value.min(*b),
            BoundType::GreaterThan(b) => value.max(b.next_up()),
            BoundType::GreaterEq(b) => value.max(*b),
            BoundType::Equal(target, _) => *target,
            BoundType::InRange(lo, hi) => {
                // `f32::clamp` panics if `lo > hi` or either is NaN.
                // `ConstraintBuilder::build` rejects that combination, but
                // `BoundType` itself derives `Deserialize` and can still
                // reach here with a degenerate range from untrusted data;
                // there is no meaningful region to project onto in that
                // case, so return `value` unchanged rather than panicking.
                if lo.is_finite() && hi.is_finite() && lo <= hi {
                    value.clamp(*lo, *hi)
                } else {
                    value
                }
            }
        }
    }

    /// Get the constraint name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the target dimension (if specific)
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Get the constraint weight for loss computation
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Get the bound type defining this constraint's feasibility region.
    pub fn bound(&self) -> &BoundType {
        &self.bound
    }
}

/// Rate-of-change constraint type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateType {
    /// Maximum rate of change: |dx/dt| <= max_rate
    MaxRate(f32),
    /// Rate must be in range: min_rate <= dx/dt <= max_rate
    RateRange { min_rate: f32, max_rate: f32 },
    /// Rate must be non-negative (monotonic increasing): dx/dt >= 0
    MonotonicIncreasing,
    /// Rate must be non-positive (monotonic decreasing): dx/dt <= 0
    MonotonicDecreasing,
}

/// Temporal constraint for rate-of-change limits
///
/// Tracks previous values to compute derivatives and enforce rate constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConstraint {
    name: String,
    dimension: Option<usize>,
    rate_type: RateType,
    dt: f32,
    weight: f32,
}

impl TemporalConstraint {
    /// Get constraint name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the target dimension (if specific)
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Get the time step
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// Get the constraint weight
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Check if a value transition satisfies this constraint
    pub fn check(&self, prev_value: f32, current_value: f32) -> bool {
        let rate = (current_value - prev_value) / self.dt;
        match &self.rate_type {
            RateType::MaxRate(max) => rate.abs() <= *max,
            RateType::RateRange { min_rate, max_rate } => rate >= *min_rate && rate <= *max_rate,
            RateType::MonotonicIncreasing => rate >= 0.0,
            RateType::MonotonicDecreasing => rate <= 0.0,
        }
    }

    /// Compute violation amount for rate constraint
    pub fn violation(&self, prev_value: f32, current_value: f32) -> f32 {
        let rate = (current_value - prev_value) / self.dt;
        match &self.rate_type {
            RateType::MaxRate(max) => (rate.abs() - max).max(0.0),
            RateType::RateRange { min_rate, max_rate } => {
                if rate < *min_rate {
                    min_rate - rate
                } else if rate > *max_rate {
                    rate - max_rate
                } else {
                    0.0
                }
            }
            RateType::MonotonicIncreasing => (-rate).max(0.0),
            RateType::MonotonicDecreasing => rate.max(0.0),
        }
    }

    /// Project a value to satisfy rate constraint given previous value
    pub fn project(&self, prev_value: f32, current_value: f32) -> f32 {
        let rate = (current_value - prev_value) / self.dt;
        match &self.rate_type {
            RateType::MaxRate(max) => {
                if rate.abs() <= *max {
                    current_value
                } else {
                    prev_value + rate.signum() * max * self.dt
                }
            }
            RateType::RateRange { min_rate, max_rate } => {
                let clamped_rate = rate.clamp(*min_rate, *max_rate);
                prev_value + clamped_rate * self.dt
            }
            RateType::MonotonicIncreasing => {
                if rate >= 0.0 {
                    current_value
                } else {
                    prev_value // Stay at previous value
                }
            }
            RateType::MonotonicDecreasing => {
                if rate <= 0.0 {
                    current_value
                } else {
                    prev_value // Stay at previous value
                }
            }
        }
    }
}

/// Builder for temporal constraints
#[derive(Default)]
pub struct TemporalConstraintBuilder {
    name: Option<String>,
    dimension: Option<usize>,
    rate_type: Option<RateType>,
    dt: Option<f32>,
    weight: f32,
}

impl TemporalConstraintBuilder {
    /// Create a new temporal constraint builder
    pub fn new() -> Self {
        Self {
            weight: 1.0,
            ..Default::default()
        }
    }

    /// Set the constraint name
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Set the target dimension
    pub fn dimension(mut self, dim: usize) -> Self {
        self.dimension = Some(dim);
        self
    }

    /// Set maximum rate of change constraint: |dx/dt| <= max_rate
    pub fn max_rate(mut self, max_rate: f32) -> Self {
        self.rate_type = Some(RateType::MaxRate(max_rate));
        self
    }

    /// Set rate range constraint: min_rate <= dx/dt <= max_rate
    pub fn rate_range(mut self, min_rate: f32, max_rate: f32) -> Self {
        self.rate_type = Some(RateType::RateRange { min_rate, max_rate });
        self
    }

    /// Set monotonic increasing constraint: dx/dt >= 0
    pub fn monotonic_increasing(mut self) -> Self {
        self.rate_type = Some(RateType::MonotonicIncreasing);
        self
    }

    /// Set monotonic decreasing constraint: dx/dt <= 0
    pub fn monotonic_decreasing(mut self) -> Self {
        self.rate_type = Some(RateType::MonotonicDecreasing);
        self
    }

    /// Set the time step (dt)
    pub fn dt(mut self, dt: f32) -> Self {
        self.dt = Some(dt);
        self
    }

    /// Set the constraint weight
    pub fn weight(mut self, w: f32) -> Self {
        self.weight = w;
        self
    }

    /// Build the temporal constraint
    pub fn build(self) -> LogicResult<TemporalConstraint> {
        let name = self
            .name
            .ok_or_else(|| LogicError::InvalidConstraint("name is required".into()))?;
        let rate_type = self
            .rate_type
            .ok_or_else(|| LogicError::InvalidConstraint("rate_type is required".into()))?;
        let dt = self
            .dt
            .ok_or_else(|| LogicError::InvalidConstraint("dt (time step) is required".into()))?;

        if dt <= 0.0 {
            return Err(LogicError::InvalidConstraint("dt must be positive".into()));
        }

        Ok(TemporalConstraint {
            name,
            dimension: self.dimension,
            rate_type,
            dt,
            weight: self.weight,
        })
    }
}

/// Temporal constraint checker that maintains state
#[derive(Debug, Clone)]
pub struct TemporalChecker {
    constraints: Vec<TemporalConstraint>,
    prev_values: Vec<f32>,
    initialized: bool,
}

impl TemporalChecker {
    /// Create a new temporal checker with given constraints
    pub fn new(constraints: Vec<TemporalConstraint>) -> Self {
        Self {
            constraints,
            prev_values: Vec::new(),
            initialized: false,
        }
    }

    /// Reset the checker state
    pub fn reset(&mut self) {
        self.prev_values.clear();
        self.initialized = false;
    }

    /// Check all temporal constraints for new values
    pub fn check(&mut self, values: &[f32]) -> Vec<(String, bool)> {
        if !self.initialized {
            self.prev_values = values.to_vec();
            self.initialized = true;
            return self
                .constraints
                .iter()
                .map(|c| (c.name.clone(), true))
                .collect();
        }

        let results: Vec<(String, bool)> = self
            .constraints
            .iter()
            .map(|c| {
                let result = if let Some(dim) = c.dimension() {
                    if dim < values.len() && dim < self.prev_values.len() {
                        c.check(self.prev_values[dim], values[dim])
                    } else {
                        true // Dimension out of bounds, consider satisfied
                    }
                } else {
                    // Check all dimensions
                    values
                        .iter()
                        .zip(self.prev_values.iter())
                        .all(|(&curr, &prev)| c.check(prev, curr))
                };
                (c.name.clone(), result)
            })
            .collect();

        self.prev_values = values.to_vec();
        results
    }

    /// Check all constraints and return total violation
    pub fn total_violation(&mut self, values: &[f32]) -> f32 {
        if !self.initialized {
            self.prev_values = values.to_vec();
            self.initialized = true;
            return 0.0;
        }

        let violation: f32 = self
            .constraints
            .iter()
            .map(|c| {
                let v = if let Some(dim) = c.dimension() {
                    if dim < values.len() && dim < self.prev_values.len() {
                        c.violation(self.prev_values[dim], values[dim])
                    } else {
                        0.0
                    }
                } else {
                    values
                        .iter()
                        .zip(self.prev_values.iter())
                        .map(|(&curr, &prev)| c.violation(prev, curr))
                        .sum()
                };
                v * c.weight()
            })
            .sum();

        self.prev_values = values.to_vec();
        violation
    }

    /// Project values to satisfy rate constraints
    pub fn project(&mut self, values: &[f32]) -> Vec<f32> {
        if !self.initialized {
            self.prev_values = values.to_vec();
            self.initialized = true;
            return values.to_vec();
        }

        let mut projected = values.to_vec();

        for c in &self.constraints {
            if let Some(dim) = c.dimension() {
                if dim < projected.len() && dim < self.prev_values.len() {
                    projected[dim] = c.project(self.prev_values[dim], projected[dim]);
                }
            } else {
                for i in 0..projected.len().min(self.prev_values.len()) {
                    projected[i] = c.project(self.prev_values[i], projected[i]);
                }
            }
        }

        self.prev_values = projected.clone();
        projected
    }

    /// Check if all constraints are satisfied
    pub fn all_satisfied(&mut self, values: &[f32]) -> bool {
        self.check(values).iter().all(|(_, sat)| *sat)
    }
}

/// Builder for constructing constraints
pub struct ConstraintBuilder {
    name: Option<String>,
    dimension: Option<usize>,
    bound: Option<BoundType>,
    weight: f32,
}

impl Default for ConstraintBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintBuilder {
    /// Create a new constraint builder
    pub fn new() -> Self {
        Self {
            name: None,
            dimension: None,
            bound: None,
            weight: 1.0,
        }
    }

    /// Set the constraint name
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Set the target dimension
    pub fn dimension(mut self, dim: usize) -> Self {
        self.dimension = Some(dim);
        self
    }

    /// Set less-than bound
    pub fn less_than(mut self, value: f32) -> Self {
        self.bound = Some(BoundType::LessThan(value));
        self
    }

    /// Set less-than-or-equal bound
    pub fn less_eq(mut self, value: f32) -> Self {
        self.bound = Some(BoundType::LessEq(value));
        self
    }

    /// Set greater-than bound
    pub fn greater_than(mut self, value: f32) -> Self {
        self.bound = Some(BoundType::GreaterThan(value));
        self
    }

    /// Set greater-than-or-equal bound
    pub fn greater_eq(mut self, value: f32) -> Self {
        self.bound = Some(BoundType::GreaterEq(value));
        self
    }

    /// Set equality constraint with tolerance
    pub fn equal(mut self, value: f32, tolerance: f32) -> Self {
        self.bound = Some(BoundType::Equal(value, tolerance));
        self
    }

    /// Set range constraint
    pub fn in_range(mut self, lo: f32, hi: f32) -> Self {
        self.bound = Some(BoundType::InRange(lo, hi));
        self
    }

    /// Set the constraint weight
    pub fn weight(mut self, w: f32) -> Self {
        self.weight = w;
        self
    }

    /// Build the constraint
    pub fn build(self) -> LogicResult<Constraint> {
        let name = self
            .name
            .ok_or_else(|| LogicError::InvalidConstraint("name is required".into()))?;
        let bound = self
            .bound
            .ok_or_else(|| LogicError::InvalidConstraint("bound is required".into()))?;
        validate_bound(&bound)?;

        Ok(Constraint {
            name,
            dimension: self.dimension,
            bound,
            weight: self.weight,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression (finding 124): `Constraint::project` on a strict bound
    /// must return a point `Constraint::check` actually accepts. The old
    /// `b - f32::EPSILON` / `b + f32::EPSILON` nudge silently became a no-op
    /// once `b`'s own ULP exceeded `f32::EPSILON` (any `|b| >= 2.0`).
    #[test]
    fn test_project_strict_bound_is_accepted_by_check() {
        let bounds = [
            1e-3_f32, 1.0, 2.0, 10.0, 1e3, 1e6, 1e9, -1e-3, -1.0, -10.0, -1e6, -1e9,
        ];
        for &b in &bounds {
            let lt = ConstraintBuilder::new()
                .name("lt")
                .less_than(b)
                .build()
                .expect("valid less_than bound");
            let projected = lt.project(b + 5.0);
            assert!(
                lt.check(projected),
                "project(b+5)={projected} not accepted by check() for LessThan({b})"
            );
            assert!(
                projected < b,
                "projected {projected} should stay below bound {b}"
            );

            let gt = ConstraintBuilder::new()
                .name("gt")
                .greater_than(b)
                .build()
                .expect("valid greater_than bound");
            let projected = gt.project(b - 5.0);
            assert!(
                gt.check(projected),
                "project(b-5)={projected} not accepted by check() for GreaterThan({b})"
            );
            assert!(
                projected > b,
                "projected {projected} should stay above bound {b}"
            );
        }
    }

    /// `project` must not panic even on a non-finite bound smuggled in past
    /// the builder (`BoundType` derives `Deserialize`, so a serde round trip
    /// can construct one directly).
    #[test]
    fn test_project_does_not_panic_on_nonfinite_bound() {
        for &b in &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let lt = Constraint {
                name: "lt".to_string(),
                dimension: None,
                bound: BoundType::LessThan(b),
                weight: 1.0,
            };
            let _ = lt.project(0.0); // must return, not panic

            let gt = Constraint {
                name: "gt".to_string(),
                dimension: None,
                bound: BoundType::GreaterThan(b),
                weight: 1.0,
            };
            let _ = gt.project(0.0);

            let range = Constraint {
                name: "range".to_string(),
                dimension: None,
                bound: BoundType::InRange(b, b),
                weight: 1.0,
            };
            let _ = range.project(0.0);
        }
    }

    /// Regression (finding 136): an inverted range must be rejected at
    /// `build()` time, not accepted and then panic later inside
    /// `Constraint::project`'s `f32::clamp` call.
    #[test]
    fn test_build_rejects_inverted_range() {
        let err = ConstraintBuilder::new()
            .name("bad")
            .in_range(5.0, 1.0)
            .build();
        assert!(matches!(err, Err(LogicError::InvalidConstraint(_))));
    }

    #[test]
    fn test_build_rejects_negative_tolerance() {
        let err = ConstraintBuilder::new()
            .name("bad")
            .equal(0.0, -1.0)
            .build();
        assert!(matches!(err, Err(LogicError::InvalidConstraint(_))));
    }

    #[test]
    fn test_build_rejects_nan_bound() {
        let err = ConstraintBuilder::new()
            .name("bad")
            .less_than(f32::NAN)
            .build();
        assert!(matches!(err, Err(LogicError::InvalidConstraint(_))));

        let err = ConstraintBuilder::new()
            .name("bad")
            .in_range(f32::NAN, 1.0)
            .build();
        assert!(matches!(err, Err(LogicError::InvalidConstraint(_))));

        let err = ConstraintBuilder::new()
            .name("bad")
            .equal(f32::NAN, 1.0)
            .build();
        assert!(matches!(err, Err(LogicError::InvalidConstraint(_))));
    }

    #[test]
    fn test_build_rejects_infinite_bound() {
        let err = ConstraintBuilder::new()
            .name("bad")
            .less_than(f32::INFINITY)
            .build();
        assert!(matches!(err, Err(LogicError::InvalidConstraint(_))));

        let err = ConstraintBuilder::new()
            .name("bad")
            .in_range(f32::NEG_INFINITY, 1.0)
            .build();
        assert!(matches!(err, Err(LogicError::InvalidConstraint(_))));
    }

    #[test]
    fn test_build_accepts_valid_bounds() {
        assert!(ConstraintBuilder::new()
            .name("ok")
            .in_range(1.0, 5.0)
            .build()
            .is_ok());
        // A zero-tolerance equality is a valid single-point interval.
        assert!(ConstraintBuilder::new()
            .name("ok")
            .equal(2.0, 0.0)
            .build()
            .is_ok());
        assert!(ConstraintBuilder::new()
            .name("ok")
            .less_than(10.0)
            .build()
            .is_ok());
        // A degenerate but *closed* single-point range remains valid.
        assert!(ConstraintBuilder::new()
            .name("ok")
            .in_range(3.0, 3.0)
            .build()
            .is_ok());
    }
}
