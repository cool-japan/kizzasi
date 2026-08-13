//! Constraint definitions and builders
//!
//! Provides constraint types for signal bounds and composition operators
//! for building complex constraint expressions, including temporal constraints
//! for rate-of-change limits.

mod basic;
mod geometric;
mod hysteresis;
mod linear;
mod ltl;
mod nonlinear;
mod quadratic;
mod sliding_window;
mod soft_hard;
mod stl;

// Re-export all public types from basic module
pub use basic::{
    BoundType, ComposedConstraint, Constraint, ConstraintBuilder, LogicalOperator, RateType,
    TemporalChecker, TemporalConstraint, TemporalConstraintBuilder,
};

// Crate-internal only: `BoundInterval` is `pub(crate)` in `basic`, but the
// `basic` module itself is private, so `constraint_analysis` (a sibling
// module) needs this re-export to reach it for exact interval-arithmetic
// feasibility/dependency analysis.
pub(crate) use basic::BoundInterval;

// Re-export from linear module
pub use linear::{AffineEquality, LinearConstraint, LinearConstraintSet, LinearConstraintType};

// Re-export from quadratic module
pub use quadratic::{QuadraticConstraint, QuadraticConstraintSet, QuadraticConstraintType};

// Re-export from nonlinear module
pub use nonlinear::{NonlinearConstraint, NonlinearConstraintType};

// Re-export from geometric module
pub use geometric::{GeometricSet, SetMembershipConstraint};

// Re-export from soft_hard module
pub use soft_hard::{
    ConstraintMode, ConstraintSet, PenaltyFunction, SoftHardConstraint, ViolationComputable,
};

// Re-export from sliding_window module
pub use sliding_window::{SlidingWindowChecker, SlidingWindowConstraint, SlidingWindowFn};

// Re-export from ltl module
pub use ltl::{LTLChecker, LTLFormula, LTLOperator};

// Re-export from hysteresis module
pub use hysteresis::{HysteresisChecker, HysteresisConstraint};

// Re-export from stl module
pub use stl::{OnlineSTLMonitor, STLFormula, STLMonitor, Signal, TimeInterval};
