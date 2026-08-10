//! # kizzasi-logic
//!
//! TensorLogic bridge for Kizzasi - constraint enforcement and safety guardrails.
//!
//! This module converts logical rules into:
//! 1. **Training Loss**: Penalize the model during training if it violates physics
//! 2. **Inference Guards**: Runtime checks to reject "hallucinated" physics
//!
//! ## COOLJAPAN Ecosystem
//!
//! This crate integrates with `tensorlogic` for neuro-symbolic constraint
//! definitions and uses `scirs2-core` for all array operations.

mod advanced_constraints;
mod advanced_projection;
mod approximate_satisfaction;
pub mod compiler;
mod constraint;
mod constraint_analysis;
mod constraint_propagation;
mod constraint_repair;
mod constraint_sensitivity;
mod decomposition;
mod differential_constraints;
pub mod distributed_admm;
mod error;
mod gpu_acceleration;
mod guardrail;
mod incremental_solver;
pub mod logic_prog;
mod mpc;
mod multiobjective;
mod online_learning;
pub mod parallel_advanced;
pub mod parallel_solver;
mod performance;
mod projection;
#[cfg(feature = "qp-solver")]
mod qp_solver;
mod tensorlogic_integration;
mod timevarying;
mod training;
mod violation_explanation;
mod visualization;

pub use advanced_constraints::{
    AmbiguitySet, CVaRConstraint, ChanceConstraint, ChanceConstraintMethod,
    DistributionallyRobustConstraint, DivergenceType, RobustConstraint, RobustnessApproach,
    UncertaintySet,
};
pub use advanced_projection::{AugmentedLagrangian, DykstraProjection, GradientProjection};
pub use approximate_satisfaction::{
    AnytimeSolver, ApproximateSolution, BoundedErrorSolver, HierarchicalConstraint,
    HierarchicalRelaxation,
};
pub use compiler::TlExprCompiler;
pub use constraint::{
    AffineEquality, BoundType, ComposedConstraint, Constraint, ConstraintBuilder, ConstraintMode,
    ConstraintSet, GeometricSet, HysteresisChecker, HysteresisConstraint, LTLChecker, LTLFormula,
    LTLOperator, LinearConstraint, LinearConstraintSet, LinearConstraintType, LogicalOperator,
    NonlinearConstraint, NonlinearConstraintType, OnlineSTLMonitor, PenaltyFunction,
    QuadraticConstraint, QuadraticConstraintSet, QuadraticConstraintType, RateType, STLFormula,
    STLMonitor, SetMembershipConstraint, Signal, SlidingWindowChecker, SlidingWindowConstraint,
    SlidingWindowFn, SoftHardConstraint, TemporalChecker, TemporalConstraint,
    TemporalConstraintBuilder, TimeInterval, ViolationComputable,
};
pub use constraint_analysis::{
    validate_constraint_set, ConsistencyAnalysis, ConstraintConsistencyChecker,
};
pub use constraint_propagation::{
    BacktrackingSearch, DiscreteConstraint, Domain, ForwardChecker, VarId, AC3, CSP,
};
pub use constraint_repair::{
    ConflictResolver, ConstraintRepairer, IISFinder, RepairResult, RepairStrategy,
};
pub use constraint_sensitivity::{
    ConstraintSensitivityAnalyzer, LocalSensitivity, SensitivityAnalysis,
};
pub use decomposition::{
    block_utils, ADMMConfig, BendersConfig, BendersCut, BendersDecomposition,
    BendersIterationResult, Block, BlockCoordinateDescent, ConsensusADMM, DecompositionError,
    DecompositionLevel, DecompositionResult, DualDecomposition, HierarchicalDecomposition,
    Subproblem,
};
pub use differential_constraints::{
    DerivativeConstraint, DerivativeOrder, DifferentialAlgebraicConstraint,
    DifferentialConstraintSet, IntegralConstraint, PathIntegralConstraint,
};
pub use error::{LogicError, LogicResult};
pub use gpu_acceleration::{GPUConstraintChecker, GPUGradientComputer, GPUProjector};
pub use guardrail::{Guardrail, GuardrailSet};
pub use incremental_solver::{
    ConstraintChange, ConstraintChangeDetector, IncrementalSolver, IncrementalState,
};
pub use mpc::{
    DynamicsModel, LinearDynamics, MPCConfig, MPCController, MPCCost, MPCSolution, QuadraticCost,
};
pub use multiobjective::{
    EpsilonConstraintMethod, HypervolumeIndicator, MultiObjectiveOptimizer, MultiObjectiveSolution,
    WeightedSumMethod,
};
pub use online_learning::{
    ActiveConstraintBoundaryLearner, AnomalyBasedConstraintDiscovery, FeedbackConstraintTuner,
    OnlineConstraintLearner, OnlineLearningSystem,
};
pub use performance::{
    AdaptiveConstraintOrder, BatchConstraintChecker, CacheStats, LazyConstraintEvaluator,
    ParallelConstraintChecker, VectorizedConstraints,
};
pub use projection::ConstrainedProjection;
#[cfg(feature = "qp-solver")]
pub use qp_solver::QPSolver;
pub use tensorlogic_integration::{
    tl_const, tl_var, ConstraintLearner, ConstraintSynthesizer, ConstraintTemplate, TLExpr,
    TLExprEvaluator, TlTerm, TypeAnnotation,
};
pub use timevarying::{
    ActivationFunction, ConstraintInterpolator, ConstraintParams, InterpolationMode,
    ParameterUpdate, PredictiveConstraintAdapter, StateDependentConstraint, TimeVaryingConstraint,
    TimeVaryingConstraintSet,
};
pub use training::{
    AdaptiveWeighting, BarrierMethod, ConstraintAwareLoss, DifferentiableProjection,
    LagrangianRelaxation, PenaltyMethod,
};
pub use violation_explanation::{
    CounterfactualAnalyzer, MinimalViolatingSubsetFinder, ViolationAttributionAnalyzer,
    ViolationExplainer, ViolationExplanation,
};
pub use visualization::{
    render_constraint_network, render_violation_heatmap, Colormap, ConstraintInspector,
    ConstraintNetworkEdge, ConstraintNetworkNode, ConstraintPlotter, ConstraintReport,
    ConstraintTimeSeries, InspectionResult, PlotConfig, SvgBuilder, ViolationStats,
};

pub use compiler::{CompiledConstraint, ConstraintExpr, ConstraintProgram, Opcode};
pub use logic_prog::{Atom, DatalogEngine, Rule, SignalFactBridge, Term};

pub use distributed_admm::{
    AdmmConfig, AdmmError, AdmmResult, ConsensusAdmm, DistributedAdmm, LassoSubproblem,
    ProjectionSubproblem, QuadraticSubproblem,
};

pub use parallel_solver::{
    BoxConstraint, ConstraintGraph, FastConstraint, HyperplaneConstraint,
    IncrementalParallelSolver, L2BallConstraint, ParallelConfig, ParallelFeasibilityChecker,
    PropagationResult, SimdConstraintEvaluator, SimplexConstraint, SolverResult,
};

// Re-export scirs2-core types
pub use scirs2_core::ndarray::Array1;

/// Trait for constrained inference
pub trait ConstrainedInference {
    /// Apply constraints to a prediction, projecting it onto the valid manifold
    fn constrain(&self, prediction: &Array1<f32>) -> LogicResult<Array1<f32>>;

    /// Check if a prediction satisfies all constraints
    fn validate(&self, prediction: &Array1<f32>) -> bool;

    /// Compute constraint violation loss for training
    fn violation_loss(&self, prediction: &Array1<f32>) -> f32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_builder() {
        let constraint = ConstraintBuilder::new()
            .name("max_velocity")
            .less_than(10.0)
            .build();

        assert!(constraint.is_ok());
    }
}
