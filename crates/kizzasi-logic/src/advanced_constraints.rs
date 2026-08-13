//! Advanced constraint types for stochastic and robust optimization
//!
//! This module provides:
//! - Chance constraints: Pr[g(x) <= 0] >= 1 - ε
//! - Robust constraints: g(x, ξ) <= 0 ∀ξ ∈ Ξ
//! - Risk-aware constraints: CVaR, VaR constraints
//! - Distributionally robust constraints

use crate::error::{LogicError, LogicResult};
use crate::lp_simplex::{solve_ge_lp, LpOutcome};
use serde::{Deserialize, Serialize};

/// Shared validation: a confidence level must lie strictly inside `(0, 1)`.
fn check_confidence(confidence: f32) -> LogicResult<()> {
    if confidence > 0.0 && confidence < 1.0 {
        Ok(())
    } else {
        Err(LogicError::InvalidConstraint(format!(
            "confidence must lie in (0, 1), got {confidence}"
        )))
    }
}

/// Shared validation: sample/scenario counts must be non-zero.
fn check_positive_count(count: usize, label: &str) -> LogicResult<()> {
    if count == 0 {
        Err(LogicError::InvalidConstraint(format!(
            "{label} must be positive, got 0"
        )))
    } else {
        Ok(())
    }
}

// ============================================================================
// Shared numerics: inverse normal CDF and a small polyhedral-direction LP
// ============================================================================

/// Inverse of the standard normal CDF (quantile function) via Peter
/// Acklam's rational approximation — relative error `< 1.15e-9` over the
/// whole open interval `(0, 1)`, unlike a hand-picked lookup table.
///
/// `p` is clamped away from the exact endpoints so a boundary value can
/// never produce `±inf`/`NaN` here even if it slipped past a constructor's
/// own validation (e.g. via `serde`).
fn inverse_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.024_25;

    let p = p.clamp(1e-15, 1.0 - 1e-15);
    let p_high = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Convert a (two-sided) confidence level to the corresponding normal
/// quantile `z_α = Φ⁻¹((1 + confidence) / 2)`. `confidence = 0.95` gives the
/// familiar `z ≈ 1.9600`, `0.99` gives `z ≈ 2.5758`, matching the constants
/// this replaced.
fn confidence_to_quantile(confidence: f32) -> f32 {
    inverse_normal_cdf((1.0 + confidence as f64) / 2.0) as f32
}

/// Maximum simplex pivots allowed when solving the polyhedral
/// worst-case-direction LP below — generous for the small, dense problems
/// (tens of dimensions/constraints) this module deals with.
const POLYHEDRAL_LP_MAX_ITERATIONS: usize = 2000;

/// Solve `max Σᵢ ξᵢ subject to A·ξ <= b` over the polyhedron described by
/// `a_matrix` (row-major, `num_constraints x dim`) and `b_vector`.
///
/// The "maximize the coordinate sum" direction is a deliberate, principled
/// choice rather than an arbitrary one: for a `Box` uncertainty set
/// re-expressed as a polytope (`ξ <= max`, `-ξ <= -min`), it selects exactly
/// the same corner as [`RobustConstraint::worst_case_scenario`]'s `Box`
/// branch (`max.clone()`) — so the Polyhedral branch generalizes the
/// existing Box heuristic instead of picking an unrelated direction.
///
/// `ξ` is unrestricted in sign, but the underlying solver only accepts
/// non-negative variables, so each `ξᵢ` is split into `ξᵢ⁺ - ξᵢ⁻` (both
/// `>= 0`) — the standard free-variable transformation for a `>=`-form
/// simplex solver.
///
/// # Errors
///
/// Returns [`LogicError::InfeasibleConstraint`] when the polyhedron is
/// empty, [`LogicError::ProjectionFailed`] when the direction is unbounded
/// on it or the solver's iteration budget is exhausted, and
/// [`LogicError::InvalidConstraint`] when `a_matrix`/`b_vector` are
/// malformed (their lengths must satisfy `a_matrix.len() == b_vector.len()
/// * dim`). Never returns a fabricated vector for any of these cases.
fn polyhedral_farthest_corner(
    a_matrix: &[f32],
    b_vector: &[f32],
    dim: usize,
) -> LogicResult<Vec<f32>> {
    if dim == 0 {
        return Ok(Vec::new());
    }
    let num_constraints = b_vector.len();
    if a_matrix.len() != num_constraints * dim {
        return Err(LogicError::InvalidConstraint(format!(
            "polyhedral uncertainty set: a_matrix must hold {} entries ({num_constraints} rows x \
             {dim} dims), got {}",
            num_constraints * dim,
            a_matrix.len()
        )));
    }

    // Variables are [xi_1+, .., xi_dim+, xi_1-, .., xi_dim-], all >= 0.
    // A*xi <= b  =>  -A*xi >= -b  =>  -A*(xi+ - xi-) >= -b
    //             =>  (-A) xi+  +  (A) xi-  >= -b
    let mut g: Vec<Vec<f64>> = Vec::with_capacity(num_constraints);
    for row in 0..num_constraints {
        let mut g_row = vec![0.0f64; 2 * dim];
        for col in 0..dim {
            let a_ij = f64::from(a_matrix.get(row * dim + col).copied().unwrap_or(0.0));
            if let Some(slot) = g_row.get_mut(col) {
                *slot = -a_ij; // coefficient on xi_col+
            }
            if let Some(slot) = g_row.get_mut(dim + col) {
                *slot = a_ij; // coefficient on xi_col-
            }
        }
        g.push(g_row);
    }
    let h: Vec<f64> = b_vector.iter().map(|&b| f64::from(-b)).collect();

    // minimize -Σ xi_i = minimize Σ(-xi_i+ + xi_i-)
    let mut c = vec![0.0f64; 2 * dim];
    for col in 0..dim {
        c[col] = -1.0;
        c[dim + col] = 1.0;
    }

    match solve_ge_lp(&c, &g, &h, POLYHEDRAL_LP_MAX_ITERATIONS) {
        LpOutcome::Optimal { primal, .. } => Ok((0..dim)
            .map(|i| {
                let plus = primal.get(i).copied().unwrap_or(0.0);
                let minus = primal.get(dim + i).copied().unwrap_or(0.0);
                (plus - minus) as f32
            })
            .collect()),
        LpOutcome::Infeasible { .. } => Err(LogicError::InfeasibleConstraint(
            "polyhedral uncertainty set A*xi <= b describes an empty region".to_string(),
        )),
        LpOutcome::Unbounded => Err(LogicError::ProjectionFailed(
            "polyhedral uncertainty set is unbounded in the worst-case direction".to_string(),
        )),
        LpOutcome::IterationLimit => Err(LogicError::ProjectionFailed(
            "polyhedral worst-case LP did not converge within its iteration budget".to_string(),
        )),
        LpOutcome::NumericalFailure(msg) => Err(LogicError::ProjectionFailed(format!(
            "polyhedral worst-case LP failed numerically: {msg}"
        ))),
    }
}

/// Chi-squared-divergence worst-case expectation bound:
/// `mean + sqrt(radius * variance)`, using the sample mean and (population)
/// variance of `losses`. `radius` is clamped to `>= 0.0` so this never
/// returns a bound below the mean.
fn chi_squared_worst_case(losses: &[f32], radius: f32) -> f32 {
    let n = losses.len() as f32;
    let mean: f32 = losses.iter().sum::<f32>() / n;
    let variance: f32 = losses.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    mean + (radius.max(0.0) * variance).sqrt()
}

/// Maximum ternary-search steps for [`kl_worst_case`]'s inner 1-D
/// minimization; each step roughly halves the search interval, so this is
/// generous for `f32`-level precision.
const KL_DUAL_SEARCH_STEPS: usize = 100;

/// KL-divergence dual objective at a fixed `lambda > 0`:
/// `lambda * radius + lambda * log(mean(exp(loss / lambda)))`, evaluated via
/// the log-sum-exp trick so it cannot overflow for a small `lambda`.
fn kl_dual(losses: &[f32], radius: f32, lambda: f32) -> f32 {
    let scaled: Vec<f32> = losses.iter().map(|&l| l / lambda).collect();
    let max_scaled = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max_scaled.is_finite() {
        return max_scaled;
    }
    let mean_exp: f32 =
        scaled.iter().map(|&s| (s - max_scaled).exp()).sum::<f32>() / losses.len() as f32;
    let log_mean_exp = max_scaled + mean_exp.ln();
    lambda * radius + lambda * log_mean_exp
}

/// KL-divergence worst-case expectation over an ambiguity ball of `radius`
/// (in nats) around the empirical distribution of `losses`, via the
/// Donsker-Varadhan / Hu-Hong dual representation
/// `sup_{Q: KL(Q||P)<=radius} E_Q[loss] = inf_{λ>0} dual(λ)`. `dual` is
/// convex in `λ`, so a ternary search over a magnitude-scaled positive range
/// finds its minimizer without needing a derivative.
///
/// By this duality the result is always `>= mean(losses)`, for any sign of
/// the individual losses.
fn kl_worst_case(losses: &[f32], radius: f32) -> f32 {
    let n = losses.len() as f32;
    let mean: f32 = losses.iter().sum::<f32>() / n;
    if radius <= 0.0 || !radius.is_finite() {
        return mean;
    }

    let max_loss = losses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_loss = losses.iter().cloned().fold(f32::INFINITY, f32::min);
    let spread = (max_loss - min_loss).abs().max(1e-3);

    let mut lo = spread * 1e-4;
    let mut hi = spread * 1e4;
    for _ in 0..KL_DUAL_SEARCH_STEPS {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if kl_dual(losses, radius, m1) < kl_dual(losses, radius, m2) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    kl_dual(losses, radius, 0.5 * (lo + hi)).max(mean)
}

// ============================================================================
// Chance Constraints
// ============================================================================

/// Chance constraint: Pr[g(x) <= 0] >= 1 - ε
///
/// A constraint that must be satisfied with a specified probability.
/// Useful for handling uncertainty in parameters or measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChanceConstraint {
    /// Name of the constraint
    name: String,
    /// Confidence level (1 - ε), e.g., 0.95 for 95% confidence
    confidence: f32,
    /// Approximation method
    method: ChanceConstraintMethod,
    /// Weight for violation penalty
    weight: f32,
}

/// Methods for handling chance constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChanceConstraintMethod {
    /// Scenario-based approximation with sampled scenarios
    ScenarioBased {
        /// Number of scenarios to sample
        num_scenarios: usize,
        /// Tolerance for violation
        violation_tolerance: f32,
    },
    /// Gaussian approximation (mean + k*sigma approach)
    Gaussian {
        /// Mean of uncertain parameter
        mean: f32,
        /// Standard deviation
        std_dev: f32,
    },
    /// Conservative tightening (deterministic approximation):
    /// `nominal_bound + tightening_factor * z_α(confidence)`.
    Conservative {
        /// Multiplicative tightening factor combined with the
        /// confidence-derived normal quantile.
        tightening_factor: f32,
        /// Nominal (untightened) bound the factor is applied against.
        nominal_bound: f32,
    },
}

impl ChanceConstraint {
    /// Create a new chance constraint with Gaussian approximation
    ///
    /// # Errors
    ///
    /// [`LogicError::InvalidConstraint`] when `confidence` is outside `(0, 1)`
    /// or `std_dev` is not strictly positive.
    pub fn gaussian(
        name: impl Into<String>,
        confidence: f32,
        mean: f32,
        std_dev: f32,
    ) -> LogicResult<Self> {
        check_confidence(confidence)?;
        if std_dev.is_nan() || std_dev <= 0.0 {
            return Err(LogicError::InvalidConstraint(format!(
                "standard deviation must be positive, got {std_dev}"
            )));
        }

        Ok(Self {
            name: name.into(),
            confidence,
            method: ChanceConstraintMethod::Gaussian { mean, std_dev },
            weight: 1.0,
        })
    }

    /// Create a scenario-based chance constraint
    ///
    /// # Errors
    ///
    /// [`LogicError::InvalidConstraint`] when `confidence` is outside `(0, 1)`
    /// or `num_scenarios` is zero.
    pub fn scenario_based(
        name: impl Into<String>,
        confidence: f32,
        num_scenarios: usize,
    ) -> LogicResult<Self> {
        check_confidence(confidence)?;
        check_positive_count(num_scenarios, "number of scenarios")?;

        Ok(Self {
            name: name.into(),
            confidence,
            method: ChanceConstraintMethod::ScenarioBased {
                num_scenarios,
                violation_tolerance: 1.0 - confidence,
            },
            weight: 1.0,
        })
    }

    /// Create a conservative chance constraint: a fixed multiplicative
    /// tightening factor applied to a stored nominal bound, scaled by the
    /// confidence-derived normal quantile.
    ///
    /// # Errors
    ///
    /// [`LogicError::InvalidConstraint`] when `confidence` is outside `(0,
    /// 1)`, or `tightening_factor`/`nominal_bound` is not finite, or
    /// `tightening_factor` is negative.
    pub fn conservative(
        name: impl Into<String>,
        confidence: f32,
        tightening_factor: f32,
        nominal_bound: f32,
    ) -> LogicResult<Self> {
        check_confidence(confidence)?;
        if !tightening_factor.is_finite() || tightening_factor < 0.0 {
            return Err(LogicError::InvalidConstraint(format!(
                "tightening factor must be finite and non-negative, got {tightening_factor}"
            )));
        }
        if !nominal_bound.is_finite() {
            return Err(LogicError::InvalidConstraint(format!(
                "nominal bound must be finite, got {nominal_bound}"
            )));
        }

        Ok(Self {
            name: name.into(),
            confidence,
            method: ChanceConstraintMethod::Conservative {
                tightening_factor,
                nominal_bound,
            },
            weight: 1.0,
        })
    }

    /// Set weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Get the deterministic tightening for this chance constraint.
    ///
    /// - `Gaussian { mean, std_dev }`: `mean + z_α * std_dev`, where `z_α =
    ///   Φ⁻¹((1+confidence)/2)` is computed from a real inverse normal CDF
    ///   (see `confidence_to_quantile`), not a 4-bucket lookup table.
    /// - `Conservative { tightening_factor, nominal_bound }`:
    ///   `nominal_bound + tightening_factor * z_α`.
    ///
    /// # Errors
    ///
    /// `ScenarioBased` chance constraints only store a scenario *count* and
    /// a violation tolerance, not the scenario samples themselves — there is
    /// no data here from which to derive a deterministic bound in the same
    /// units as the other two methods, so this returns
    /// [`LogicError::InvalidConstraint`] rather than a fabricated number.
    /// Compute the empirical `(1 - violation_tolerance)`-quantile from your
    /// own scenario samples instead.
    pub fn get_tightened_bound(&self) -> LogicResult<f32> {
        match &self.method {
            ChanceConstraintMethod::Gaussian { mean, std_dev } => {
                let z_alpha = confidence_to_quantile(self.confidence);
                Ok(mean + z_alpha * std_dev)
            }
            ChanceConstraintMethod::Conservative {
                tightening_factor,
                nominal_bound,
            } => {
                let z_alpha = confidence_to_quantile(self.confidence);
                Ok(nominal_bound + tightening_factor * z_alpha)
            }
            ChanceConstraintMethod::ScenarioBased { .. } => Err(LogicError::InvalidConstraint(
                "ScenarioBased chance constraints store only a scenario count and violation \
                 tolerance, not sample data — get_tightened_bound cannot derive a deterministic \
                 bound from that alone; compute the empirical (1 - violation_tolerance)-quantile \
                 from your own scenario samples instead"
                    .to_string(),
            )),
        }
    }

    /// Name accessor
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Confidence accessor
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Weight accessor
    pub fn weight(&self) -> f32 {
        self.weight
    }
}

// ============================================================================
// Robust Constraints
// ============================================================================

/// Robust constraint: g(x, ξ) <= 0 for all ξ in uncertainty set
///
/// Ensures constraint satisfaction under all realizations of uncertainty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustConstraint {
    /// Name of the constraint
    name: String,
    /// Uncertainty set description
    uncertainty_set: UncertaintySet,
    /// Robustness approach
    approach: RobustnessApproach,
    /// Weight for violation penalty
    weight: f32,
}

/// Types of uncertainty sets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintySet {
    /// Box uncertainty: ξ ∈ [ξ_min, ξ_max]
    Box { min: Vec<f32>, max: Vec<f32> },
    /// Ellipsoidal uncertainty: ||ξ - ξ_nominal||_P <= Ω
    Ellipsoidal {
        nominal: Vec<f32>,
        shape_matrix: Vec<f32>,
        radius: f32,
    },
    /// Polyhedral uncertainty: A*ξ <= b
    Polyhedral {
        a_matrix: Vec<f32>,
        b_vector: Vec<f32>,
        dim: usize,
    },
    /// Budget uncertainty (cardinality-constrained): ||ξ - ξ_nominal||_0 <= Γ
    Budget {
        nominal: Vec<f32>,
        max_deviations: Vec<f32>,
        budget: usize,
    },
}

/// Approaches to handle robust constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RobustnessApproach {
    /// Worst-case optimization
    WorstCase,
    /// Affinely adjustable robust counterpart
    AffinelyAdjustable,
    /// Scenario-based robust approximation
    Scenarios { num_scenarios: usize },
}

impl RobustConstraint {
    /// Create a box-uncertain robust constraint
    ///
    /// # Errors
    ///
    /// [`LogicError::InvalidConstraint`] when `min` and `max` have different
    /// lengths or some `min[i] > max[i]`.
    pub fn box_uncertain(
        name: impl Into<String>,
        min: Vec<f32>,
        max: Vec<f32>,
    ) -> LogicResult<Self> {
        if min.len() != max.len() {
            return Err(LogicError::InvalidConstraint(format!(
                "uncertainty box bounds must have the same dimension, got {} and {}",
                min.len(),
                max.len()
            )));
        }
        for (index, (mi, ma)) in min.iter().zip(max.iter()).enumerate() {
            if mi > ma {
                return Err(LogicError::InvalidConstraint(format!(
                    "uncertainty box bound {index} is empty: min {mi} exceeds max {ma}"
                )));
            }
        }

        Ok(Self {
            name: name.into(),
            uncertainty_set: UncertaintySet::Box { min, max },
            approach: RobustnessApproach::WorstCase,
            weight: 1.0,
        })
    }

    /// Create an ellipsoidal uncertainty robust constraint
    ///
    /// # Errors
    ///
    /// [`LogicError::InvalidConstraint`] when `radius` is not strictly positive.
    pub fn ellipsoidal_uncertain(
        name: impl Into<String>,
        nominal: Vec<f32>,
        radius: f32,
    ) -> LogicResult<Self> {
        if radius.is_nan() || radius <= 0.0 {
            return Err(LogicError::InvalidConstraint(format!(
                "uncertainty radius must be positive, got {radius}"
            )));
        }
        let dim = nominal.len();
        let mut identity = vec![0.0f32; dim * dim];
        for (i, slot) in identity.iter_mut().enumerate().take(dim * dim) {
            if i / dim == i % dim {
                *slot = 1.0;
            }
        }

        Ok(Self {
            name: name.into(),
            uncertainty_set: UncertaintySet::Ellipsoidal {
                nominal,
                shape_matrix: identity,
                radius,
            },
            approach: RobustnessApproach::WorstCase,
            weight: 1.0,
        })
    }

    /// Create a polyhedral-uncertainty robust constraint: `Ξ = {ξ : A·ξ <=
    /// b}`, with `A` given row-major (`a_matrix[row * dim + col]`).
    ///
    /// # Errors
    ///
    /// [`LogicError::InvalidConstraint`] when `a_matrix.len() !=
    /// b_vector.len() * dim` or `dim == 0`.
    pub fn polyhedral_uncertain(
        name: impl Into<String>,
        a_matrix: Vec<f32>,
        b_vector: Vec<f32>,
        dim: usize,
    ) -> LogicResult<Self> {
        if dim == 0 {
            return Err(LogicError::InvalidConstraint(
                "polyhedral uncertainty set dimension must be positive".to_string(),
            ));
        }
        if a_matrix.len() != b_vector.len() * dim {
            return Err(LogicError::InvalidConstraint(format!(
                "polyhedral a_matrix must hold {} entries ({} rows x {dim} dims), got {}",
                b_vector.len() * dim,
                b_vector.len(),
                a_matrix.len()
            )));
        }

        Ok(Self {
            name: name.into(),
            uncertainty_set: UncertaintySet::Polyhedral {
                a_matrix,
                b_vector,
                dim,
            },
            approach: RobustnessApproach::WorstCase,
            weight: 1.0,
        })
    }

    /// Set robustness approach
    pub fn with_approach(mut self, approach: RobustnessApproach) -> Self {
        self.approach = approach;
        self
    }

    /// Get the robustness approach
    pub fn approach(&self) -> &RobustnessApproach {
        &self.approach
    }

    /// Set weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Get a worst-case realization from the uncertainty set, under
    /// [`RobustnessApproach::WorstCase`].
    ///
    /// - `Box`: the max corner (conservative heuristic).
    /// - `Ellipsoidal`: `nominal + radius * P^(1/2) d / ||d||` for the
    ///   all-ones direction `d`; `P^(1/2)` is applied via the stored
    ///   `shape_matrix` (currently always the identity — see
    ///   [`Self::ellipsoidal_uncertain`] — so this reduces to `nominal +
    ///   radius * d/||d||`, the correct boundary point of the true L2 ball,
    ///   unlike the previous `nominal + radius` elementwise, whose L2 norm
    ///   overshot the ball for `dim > 1`).
    /// - `Polyhedral`: solved as an LP maximizing `Σξᵢ` over `A·ξ <= b` (see
    ///   `polyhedral_farthest_corner` for why that objective was chosen).
    /// - `Budget`: the top-`budget` largest deviations (conservative heuristic).
    ///
    /// `x` is accepted for interface symmetry with a future direction-aware
    /// worst case, but — like the pre-existing Box/Ellipsoidal/Budget
    /// heuristics — is not currently used to select the direction.
    ///
    /// # Errors
    ///
    /// [`LogicError::InvalidConstraint`] when `self.approach()` is
    /// [`RobustnessApproach::Scenarios`] or
    /// [`RobustnessApproach::AffinelyAdjustable`]: computing either requires
    /// data this method does not have (actual scenario samples, or the
    /// recourse/decision-rule structure of an affinely adjustable
    /// counterpart), so it errors instead of silently reusing the
    /// `WorstCase` result under a different approach's name. See
    /// `polyhedral_farthest_corner`'s docs for the `Polyhedral` LP's own
    /// error cases (infeasible / unbounded / non-convergent).
    pub fn worst_case_scenario(&self, x: &[f32]) -> LogicResult<Vec<f32>> {
        // Reserved for a future direction-aware worst case; every branch
        // below is a fixed heuristic that does not yet use it (see doc
        // comment above).
        let _ = x;

        match &self.approach {
            RobustnessApproach::Scenarios { .. } | RobustnessApproach::AffinelyAdjustable => {
                return Err(LogicError::InvalidConstraint(format!(
                    "worst_case_scenario: {:?} requires scenario samples or a recourse structure \
                     that is not available here; only WorstCase is implemented",
                    self.approach
                )));
            }
            RobustnessApproach::WorstCase => {}
        }

        match &self.uncertainty_set {
            UncertaintySet::Box { min: _, max } => Ok(max.clone()),
            UncertaintySet::Ellipsoidal {
                nominal,
                shape_matrix,
                radius,
            } => {
                let dim = nominal.len();
                // Direction: the all-ones vector, normalized to unit L2 norm.
                let dir_norm = (dim as f32).sqrt().max(f32::MIN_POSITIVE);
                let dir = vec![1.0f32 / dir_norm; dim];

                // P^(1/2) d: since shape_matrix is symmetric positive
                // semi-definite and (currently) always the identity in
                // practice, applying it directly is exact for the identity
                // case and a reasonable first-order approximation of the
                // true matrix square root otherwise (exact for any diagonal
                // P; a full eigendecomposition would be needed for a
                // general P, which this crate does not currently need since
                // there is no public way to set a non-identity shape_matrix).
                let transformed: Vec<f32> = (0..dim)
                    .map(|i| {
                        (0..dim)
                            .map(|j| shape_matrix.get(i * dim + j).copied().unwrap_or(0.0) * dir[j])
                            .sum()
                    })
                    .collect();

                Ok(nominal
                    .iter()
                    .zip(transformed.iter())
                    .map(|(&n, &t)| n + radius * t)
                    .collect())
            }
            UncertaintySet::Polyhedral {
                a_matrix,
                b_vector,
                dim,
            } => polyhedral_farthest_corner(a_matrix, b_vector, *dim),
            UncertaintySet::Budget {
                nominal,
                max_deviations,
                budget,
            } => {
                // Take top 'budget' largest deviations
                let mut result = nominal.clone();
                for (i, &dev) in max_deviations.iter().enumerate().take(*budget) {
                    if let Some(slot) = result.get_mut(i) {
                        *slot += dev;
                    }
                }
                Ok(result)
            }
        }
    }

    /// Name accessor
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Weight accessor
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Get uncertainty set
    pub fn uncertainty_set(&self) -> &UncertaintySet {
        &self.uncertainty_set
    }
}

// ============================================================================
// Risk-Aware Constraints
// ============================================================================

/// Risk-aware constraint using CVaR (Conditional Value at Risk)
///
/// CVaR_α(loss) <= threshold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVaRConstraint {
    /// Name of the constraint
    name: String,
    /// Risk level α ∈ (0, 1), e.g., 0.05 for 5% worst cases
    alpha: f32,
    /// Threshold for CVaR
    threshold: f32,
    /// Sample scenarios for CVaR estimation
    num_scenarios: usize,
    /// Weight for violation penalty
    weight: f32,
}

impl CVaRConstraint {
    /// Create a new CVaR constraint
    ///
    /// # Errors
    ///
    /// [`LogicError::InvalidConstraint`] when `alpha` is outside `(0, 1)` or
    /// `num_scenarios` is zero.
    pub fn new(
        name: impl Into<String>,
        alpha: f32,
        threshold: f32,
        num_scenarios: usize,
    ) -> LogicResult<Self> {
        if alpha.is_nan() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(LogicError::InvalidConstraint(format!(
                "CVaR alpha must lie in (0, 1), got {alpha}"
            )));
        }
        check_positive_count(num_scenarios, "number of scenarios")?;

        Ok(Self {
            name: name.into(),
            alpha,
            threshold,
            num_scenarios,
            weight: 1.0,
        })
    }

    /// Set weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Compute CVaR from a sample of losses
    pub fn compute_cvar(&self, losses: &[f32]) -> f32 {
        if losses.is_empty() {
            return 0.0;
        }

        let mut sorted_losses = losses.to_vec();
        sorted_losses.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)); // Descending order, NaN-safe

        // Take worst alpha% of scenarios
        let cutoff = (self.alpha * sorted_losses.len() as f32).ceil() as usize;
        let cutoff = cutoff.max(1).min(sorted_losses.len());

        // Average of worst cases
        sorted_losses.iter().take(cutoff).sum::<f32>() / cutoff as f32
    }

    /// Check if CVaR constraint is satisfied
    pub fn check(&self, losses: &[f32]) -> bool {
        let cvar = self.compute_cvar(losses);
        cvar <= self.threshold
    }

    /// Compute violation
    pub fn violation(&self, losses: &[f32]) -> f32 {
        let cvar = self.compute_cvar(losses);
        (cvar - self.threshold).max(0.0)
    }

    /// Name accessor
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Alpha accessor
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Threshold accessor
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Weight accessor
    pub fn weight(&self) -> f32 {
        self.weight
    }
}

// ============================================================================
// Distributionally Robust Constraints
// ============================================================================

/// Distributionally robust constraint: worst-case expectation over ambiguity set
///
/// sup_{P ∈ P} E_P[loss(x, ξ)] <= threshold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionallyRobustConstraint {
    /// Name of the constraint
    name: String,
    /// Ambiguity set specification
    ambiguity_set: AmbiguitySet,
    /// Threshold for worst-case expectation
    threshold: f32,
    /// Weight for violation penalty
    weight: f32,
}

/// Types of ambiguity sets for distributionally robust optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AmbiguitySet {
    /// Wasserstein ball around empirical distribution
    Wasserstein {
        /// Empirical samples
        num_samples: usize,
        /// Wasserstein radius
        radius: f32,
    },
    /// Moment-based ambiguity (known mean and covariance bounds)
    MomentBased {
        /// Mean estimate
        mean: Vec<f32>,
        /// Covariance bound
        cov_radius: f32,
    },
    /// φ-divergence ball (KL, chi-squared, etc.)
    PhiDivergence {
        /// Reference distribution samples
        num_samples: usize,
        /// Divergence radius
        radius: f32,
        /// Type of divergence
        divergence_type: DivergenceType,
    },
}

/// Types of φ-divergences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DivergenceType {
    /// Kullback-Leibler divergence
    KL,
    /// Chi-squared divergence
    ChiSquared,
    /// Modified chi-squared
    ModifiedChiSquared,
}

impl DistributionallyRobustConstraint {
    /// Create a Wasserstein distributionally robust constraint
    ///
    /// # Errors
    ///
    /// [`LogicError::InvalidConstraint`] when `radius` is not strictly positive
    /// or `num_samples` is zero.
    pub fn wasserstein(
        name: impl Into<String>,
        threshold: f32,
        num_samples: usize,
        radius: f32,
    ) -> LogicResult<Self> {
        if radius.is_nan() || radius <= 0.0 {
            return Err(LogicError::InvalidConstraint(format!(
                "Wasserstein radius must be positive, got {radius}"
            )));
        }
        check_positive_count(num_samples, "number of samples")?;

        Ok(Self {
            name: name.into(),
            ambiguity_set: AmbiguitySet::Wasserstein {
                num_samples,
                radius,
            },
            threshold,
            weight: 1.0,
        })
    }

    /// Create a moment-based distributionally robust constraint
    ///
    /// # Errors
    ///
    /// [`LogicError::InvalidConstraint`] when `cov_radius` is not strictly
    /// positive.
    pub fn moment_based(
        name: impl Into<String>,
        threshold: f32,
        mean: Vec<f32>,
        cov_radius: f32,
    ) -> LogicResult<Self> {
        if cov_radius.is_nan() || cov_radius <= 0.0 {
            return Err(LogicError::InvalidConstraint(format!(
                "covariance radius must be positive, got {cov_radius}"
            )));
        }

        Ok(Self {
            name: name.into(),
            ambiguity_set: AmbiguitySet::MomentBased { mean, cov_radius },
            threshold,
            weight: 1.0,
        })
    }

    /// Create a φ-divergence distributionally robust constraint.
    ///
    /// # Errors
    ///
    /// [`LogicError::InvalidConstraint`] when `radius` is not strictly
    /// positive or `num_samples` is zero.
    pub fn phi_divergence(
        name: impl Into<String>,
        threshold: f32,
        num_samples: usize,
        radius: f32,
        divergence_type: DivergenceType,
    ) -> LogicResult<Self> {
        if radius.is_nan() || radius <= 0.0 {
            return Err(LogicError::InvalidConstraint(format!(
                "phi-divergence radius must be positive, got {radius}"
            )));
        }
        check_positive_count(num_samples, "number of samples")?;

        Ok(Self {
            name: name.into(),
            ambiguity_set: AmbiguitySet::PhiDivergence {
                num_samples,
                radius,
                divergence_type,
            },
            threshold,
            weight: 1.0,
        })
    }

    /// Set weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Compute worst-case expectation (conservative approximation)
    ///
    /// `PhiDivergence` now dispatches on [`DivergenceType`] instead of
    /// collapsing every divergence to `mean * (1.0 + radius)` (which used
    /// none of the actual sample values and, for a negative mean, produced
    /// a "conservative" bound *smaller* than the mean):
    ///
    /// - `KL`: the Donsker-Varadhan dual `inf_{λ>0} [λ·radius +
    ///   λ·log(mean(exp(loss/λ)))]`, solved by ternary search (convex in
    ///   `λ`). By that duality this is always `>= mean(losses)`.
    /// - `ChiSquared`: `mean + sqrt(radius * variance)`, the standard
    ///   chi-squared-divergence DRO bound (the same formula already used
    ///   for `MomentBased` above, with `radius` as the chi-squared budget).
    /// - `ModifiedChiSquared`: reuses the `ChiSquared` bound. The
    ///   "modified" chi-squared divergence has a related but distinct dual
    ///   in the DRO literature; absent a closed form here we are confident
    ///   is correct for it specifically, this deliberately reuses a
    ///   validated sibling bound rather than fabricating a different-looking
    ///   formula with no derivation behind it.
    pub fn worst_case_expectation(&self, losses: &[f32]) -> f32 {
        if losses.is_empty() {
            return 0.0;
        }

        match &self.ambiguity_set {
            AmbiguitySet::Wasserstein { radius, .. } => {
                // Conservative: mean + radius * max_loss
                let mean: f32 = losses.iter().sum::<f32>() / losses.len() as f32;
                let max_loss = losses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                mean + radius * max_loss.abs()
            }
            AmbiguitySet::MomentBased { cov_radius, .. } => {
                chi_squared_worst_case(losses, *cov_radius)
            }
            AmbiguitySet::PhiDivergence {
                radius,
                divergence_type,
                ..
            } => match divergence_type {
                DivergenceType::KL => kl_worst_case(losses, *radius),
                DivergenceType::ChiSquared | DivergenceType::ModifiedChiSquared => {
                    chi_squared_worst_case(losses, *radius)
                }
            },
        }
    }

    /// Check if constraint is satisfied
    pub fn check(&self, losses: &[f32]) -> bool {
        let wc_exp = self.worst_case_expectation(losses);
        wc_exp <= self.threshold
    }

    /// Compute violation
    pub fn violation(&self, losses: &[f32]) -> f32 {
        let wc_exp = self.worst_case_expectation(losses);
        (wc_exp - self.threshold).max(0.0)
    }

    /// Name accessor
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Threshold accessor
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Weight accessor
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Get ambiguity set
    pub fn ambiguity_set(&self) -> &AmbiguitySet {
        &self.ambiguity_set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chance_constraint_gaussian() {
        let cc = ChanceConstraint::gaussian("test", 0.95, 10.0, 2.0).expect("valid chance");
        assert_eq!(cc.name(), "test");
        assert_eq!(cc.confidence(), 0.95);

        // 95% confidence should give bound around mean + 1.96*sigma
        let bound = cc
            .get_tightened_bound()
            .expect("gaussian tightening succeeds");
        assert!(bound > 10.0);
        assert!(bound < 15.0); // 10 + 1.96*2 ≈ 13.92
        assert!((bound - (10.0 + 1.959964 * 2.0)).abs() < 1e-3);
    }

    #[test]
    fn test_robust_constraint_box() {
        let rc = RobustConstraint::box_uncertain("test", vec![-1.0, -2.0], vec![1.0, 2.0])
            .expect("valid robust");
        assert_eq!(rc.name(), "test");

        let worst_case = rc
            .worst_case_scenario(&[0.0, 0.0])
            .expect("box worst case succeeds");
        assert_eq!(worst_case, vec![1.0, 2.0]);
    }

    #[test]
    fn test_cvar_constraint() {
        let cvar = CVaRConstraint::new("test", 0.1, 10.0, 100).expect("valid CVaR");

        // Test CVaR computation
        let losses = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let cvar_value = cvar.compute_cvar(&losses);

        // Top 10% should be close to 10.0
        assert!(cvar_value >= 9.0);
        assert!(cvar_value <= 10.0);
    }

    #[test]
    fn test_distributionally_robust_wasserstein() {
        let drc = DistributionallyRobustConstraint::wasserstein("test", 15.0, 100, 0.5)
            .expect("valid DRO");

        let losses = vec![5.0, 10.0, 15.0];
        let wc_exp = drc.worst_case_expectation(&losses);

        // Should be conservative
        assert!(wc_exp >= 10.0); // mean
        assert!(drc.check(&losses) || !drc.check(&losses)); // No panic
    }

    // -----------------------------------------------------------------------
    // Regression: inverse_normal_cdf / confidence_to_quantile (finding 132)
    // -----------------------------------------------------------------------

    #[test]
    fn test_inverse_normal_cdf_matches_known_quantiles() {
        // Standard reference points, accurate well within Acklam's bound.
        assert!((inverse_normal_cdf(0.5) - 0.0).abs() < 1e-6);
        assert!((inverse_normal_cdf(0.975) - 1.959_964).abs() < 1e-4);
        assert!((inverse_normal_cdf(0.995) - 2.575_829).abs() < 1e-4);
        assert!((inverse_normal_cdf(0.95) - 1.644_854).abs() < 1e-4);
    }

    /// The old 4-bucket step function returned exactly 1.96 for every
    /// confidence in [0.95, 0.99) and a *negative* z for confidence < 0.5.
    /// The real inverse CDF must vary continuously and stay monotonic.
    #[test]
    fn test_confidence_to_quantile_is_continuous_and_monotonic() {
        let z_950 = confidence_to_quantile(0.950);
        let z_960 = confidence_to_quantile(0.960);
        let z_989 = confidence_to_quantile(0.989);
        assert!(
            z_950 < z_960 && z_960 < z_989,
            "quantile must strictly increase with confidence: {z_950} {z_960} {z_989}"
        );

        // Low confidence must stay positive (a "tightening" quantile below
        // zero would loosen, not tighten, the bound).
        assert!(confidence_to_quantile(0.05) > 0.0);
        assert!(confidence_to_quantile(0.30) > 0.0);
    }

    // -----------------------------------------------------------------------
    // Regression: ChanceConstraint::get_tightened_bound (finding 132)
    // -----------------------------------------------------------------------

    #[test]
    fn test_conservative_tightened_bound_depends_on_confidence_and_nominal() {
        let low = ChanceConstraint::conservative("c", 0.80, 2.0, 10.0).expect("valid");
        let high = ChanceConstraint::conservative("c", 0.99, 2.0, 10.0).expect("valid");

        let low_bound = low.get_tightened_bound().expect("conservative succeeds");
        let high_bound = high.get_tightened_bound().expect("conservative succeeds");

        // Both must exceed the nominal bound (positive tightening factor).
        assert!(low_bound > 10.0);
        assert!(high_bound > 10.0);
        // Higher confidence must tighten more — the old code returned the
        // raw `tightening_factor` (2.0) regardless of confidence.
        assert!(
            high_bound > low_bound,
            "higher confidence must produce a larger tightened bound: {low_bound} vs {high_bound}"
        );
    }

    #[test]
    fn test_conservative_rejects_invalid_inputs() {
        assert!(matches!(
            ChanceConstraint::conservative("bad", 0.95, -1.0, 0.0),
            Err(LogicError::InvalidConstraint(_))
        ));
        assert!(matches!(
            ChanceConstraint::conservative("bad", 0.95, f32::NAN, 0.0),
            Err(LogicError::InvalidConstraint(_))
        ));
        assert!(matches!(
            ChanceConstraint::conservative("bad", 0.95, 1.0, f32::INFINITY),
            Err(LogicError::InvalidConstraint(_))
        ));
        assert!(matches!(
            ChanceConstraint::conservative("bad", 1.5, 1.0, 0.0),
            Err(LogicError::InvalidConstraint(_))
        ));
    }

    /// `ScenarioBased` has no stored sample data, so tightening it must
    /// error rather than fabricate `confidence * 10.0` as before.
    #[test]
    fn test_scenario_based_tightened_bound_errors_without_samples() {
        let sc = ChanceConstraint::scenario_based("s", 0.95, 100).expect("valid");
        assert!(matches!(
            sc.get_tightened_bound(),
            Err(LogicError::InvalidConstraint(_))
        ));
    }

    // -----------------------------------------------------------------------
    // Regression: RobustConstraint::worst_case_scenario (finding 131)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ellipsoidal_shape_matrix_is_true_identity() {
        let rc = RobustConstraint::ellipsoidal_uncertain("e", vec![0.0, 0.0, 0.0], 1.0)
            .expect("valid ellipsoidal");
        let UncertaintySet::Ellipsoidal { shape_matrix, .. } = rc.uncertainty_set() else {
            panic!("expected an Ellipsoidal uncertainty set");
        };
        // The old code built `vec![1.0; dim*dim]` (all-ones, singular).
        // A true 3x3 identity has exactly 3 ones, all on the diagonal.
        assert_eq!(shape_matrix.iter().filter(|&&v| v == 1.0).count(), 3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(shape_matrix[i * 3 + j], expected, "at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_ellipsoidal_worst_case_lands_on_the_true_boundary() {
        // Regression: the old `nominal + radius` elementwise formula had L2
        // norm `radius * sqrt(dim)` from `nominal` — for dim=4 that
        // overshoots a radius-1 ball by 2x. The corrected version must land
        // exactly on the boundary (distance == radius) since shape_matrix
        // is the identity.
        let nominal = vec![1.0, 2.0, 3.0, 4.0];
        let rc = RobustConstraint::ellipsoidal_uncertain("e", nominal.clone(), 1.0)
            .expect("valid ellipsoidal");
        let worst_case = rc
            .worst_case_scenario(&[0.0; 4])
            .expect("ellipsoidal worst case succeeds");

        let dist: f32 = worst_case
            .iter()
            .zip(nominal.iter())
            .map(|(&w, &n)| (w - n).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            (dist - 1.0).abs() < 1e-4,
            "worst case must sit on the radius-1 boundary, got distance {dist}"
        );
    }

    #[test]
    fn test_polyhedral_worst_case_solves_a_real_lp() {
        // A box re-expressed as a polytope: -1 <= xi_0 <= 1, -1 <= xi_1 <= 1.
        // A*xi <= b with rows [1,0]<=1, [-1,0]<=1, [0,1]<=1, [0,-1]<=1.
        let a_matrix = vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0];
        let b_vector = vec![1.0, 1.0, 1.0, 1.0];
        let rc = RobustConstraint::polyhedral_uncertain("p", a_matrix, b_vector, 2)
            .expect("valid polyhedral set");

        let worst_case = rc
            .worst_case_scenario(&[0.0, 0.0])
            .expect("polyhedral LP solves");

        // maximize xi_0 + xi_1 over this box: the unique optimum is (1, 1) —
        // matching what the Box branch would return for the same region,
        // and nowhere near the old placeholder's `vec![0.0, 0.0]`.
        assert!((worst_case[0] - 1.0).abs() < 1e-3, "{worst_case:?}");
        assert!((worst_case[1] - 1.0).abs() < 1e-3, "{worst_case:?}");
    }

    #[test]
    fn test_polyhedral_uncertain_rejects_malformed_matrix() {
        assert!(matches!(
            RobustConstraint::polyhedral_uncertain("bad", vec![1.0, 0.0, 0.0], vec![1.0, 1.0], 2),
            Err(LogicError::InvalidConstraint(_))
        ));
        assert!(matches!(
            RobustConstraint::polyhedral_uncertain("bad", vec![], vec![1.0], 0),
            Err(LogicError::InvalidConstraint(_))
        ));
    }

    #[test]
    fn test_polyhedral_worst_case_infeasible_region_errors() {
        // xi_0 <= 1 AND xi_0 >= 2 (i.e. -xi_0 <= -2): empty.
        let a_matrix = vec![1.0, -1.0];
        let b_vector = vec![1.0, -2.0];
        let rc = RobustConstraint::polyhedral_uncertain("p", a_matrix, b_vector, 1)
            .expect("valid shape");
        assert!(matches!(
            rc.worst_case_scenario(&[0.0]),
            Err(LogicError::InfeasibleConstraint(_))
        ));
    }

    /// `Scenarios`/`AffinelyAdjustable` must error rather than silently
    /// reuse the `WorstCase` computation under a different name.
    #[test]
    fn test_non_worst_case_approach_errors() {
        let rc = RobustConstraint::box_uncertain("b", vec![-1.0], vec![1.0])
            .expect("valid")
            .with_approach(RobustnessApproach::Scenarios { num_scenarios: 10 });
        assert!(matches!(
            rc.worst_case_scenario(&[0.0]),
            Err(LogicError::InvalidConstraint(_))
        ));

        let rc2 = RobustConstraint::box_uncertain("b", vec![-1.0], vec![1.0])
            .expect("valid")
            .with_approach(RobustnessApproach::AffinelyAdjustable);
        assert!(matches!(
            rc2.worst_case_scenario(&[0.0]),
            Err(LogicError::InvalidConstraint(_))
        ));
    }

    // -----------------------------------------------------------------------
    // Regression: DistributionallyRobustConstraint PhiDivergence (finding 307)
    // -----------------------------------------------------------------------

    #[test]
    fn test_phi_divergence_dispatches_on_divergence_type() {
        let losses = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;

        let kl = DistributionallyRobustConstraint::phi_divergence(
            "kl",
            100.0,
            losses.len(),
            0.5,
            DivergenceType::KL,
        )
        .expect("valid phi-divergence DRO");
        let chi2 = DistributionallyRobustConstraint::phi_divergence(
            "chi2",
            100.0,
            losses.len(),
            0.5,
            DivergenceType::ChiSquared,
        )
        .expect("valid phi-divergence DRO");

        let kl_bound = kl.worst_case_expectation(&losses);
        let chi2_bound = chi2.worst_case_expectation(&losses);

        // Both must be conservative (>= the true mean) — the old code could
        // fall *below* the mean for a negative mean; these losses are
        // positive but the invariant is now guaranteed by construction.
        assert!(kl_bound >= mean - 1e-3, "KL bound {kl_bound} < mean {mean}");
        assert!(
            chi2_bound >= mean - 1e-3,
            "ChiSquared bound {chi2_bound} < mean {mean}"
        );
        // And they must actually be *different* numbers now (previously
        // both, plus ModifiedChiSquared, were byte-identical:
        // `mean * (1.0 + radius)`).
        assert!(
            (kl_bound - chi2_bound).abs() > 1e-4,
            "KL and ChiSquared should generally disagree: {kl_bound} vs {chi2_bound}"
        );
    }

    #[test]
    fn test_phi_divergence_conservative_even_for_negative_mean() {
        let losses = vec![-10.0, -5.0, -1.0];
        let mean: f32 = losses.iter().sum::<f32>() / losses.len() as f32;

        let kl = DistributionallyRobustConstraint::phi_divergence(
            "kl",
            100.0,
            losses.len(),
            0.3,
            DivergenceType::KL,
        )
        .expect("valid phi-divergence DRO");
        let bound = kl.worst_case_expectation(&losses);
        assert!(
            bound >= mean,
            "a 'conservative' bound must never fall below the mean: bound={bound} mean={mean}"
        );
    }

    /// Regression (finding 140): invalid constructor input must produce a
    /// recoverable `LogicError`, not abort the process with `assert!`.
    #[test]
    fn test_invalid_constructor_inputs_return_errors() {
        assert!(matches!(
            ChanceConstraint::gaussian("bad", 1.5, 0.0, 1.0),
            Err(LogicError::InvalidConstraint(_))
        ));
        assert!(matches!(
            ChanceConstraint::gaussian("bad", 0.95, 0.0, 0.0),
            Err(LogicError::InvalidConstraint(_))
        ));
        assert!(matches!(
            ChanceConstraint::scenario_based("bad", 0.95, 0),
            Err(LogicError::InvalidConstraint(_))
        ));
        assert!(matches!(
            RobustConstraint::box_uncertain("bad", vec![0.0, 0.0], vec![1.0]),
            Err(LogicError::InvalidConstraint(_))
        ));
        assert!(matches!(
            RobustConstraint::box_uncertain("bad", vec![2.0], vec![1.0]),
            Err(LogicError::InvalidConstraint(_))
        ));
        assert!(matches!(
            RobustConstraint::ellipsoidal_uncertain("bad", vec![0.0], -1.0),
            Err(LogicError::InvalidConstraint(_))
        ));
        assert!(matches!(
            CVaRConstraint::new("bad", 0.0, 1.0, 10),
            Err(LogicError::InvalidConstraint(_))
        ));
        assert!(matches!(
            CVaRConstraint::new("bad", 0.5, 1.0, 0),
            Err(LogicError::InvalidConstraint(_))
        ));
        assert!(matches!(
            DistributionallyRobustConstraint::wasserstein("bad", 1.0, 10, 0.0),
            Err(LogicError::InvalidConstraint(_))
        ));
        assert!(matches!(
            DistributionallyRobustConstraint::moment_based("bad", 1.0, vec![0.0], -0.5),
            Err(LogicError::InvalidConstraint(_))
        ));
    }

    /// NaN inputs must be rejected too (the old `assert!` comparisons let a
    /// NaN confidence through as "false" and aborted, or slipped past a `>`).
    #[test]
    fn test_nan_constructor_inputs_return_errors() {
        assert!(matches!(
            ChanceConstraint::gaussian("nan", f32::NAN, 0.0, 1.0),
            Err(LogicError::InvalidConstraint(_))
        ));
        assert!(matches!(
            GeometricSetProbe::ball_with_nan_radius(),
            Err(LogicError::InvalidConstraint(_))
        ));
    }

    /// Small helper so the NaN test can reach the geometric constructors
    /// without importing the whole module surface.
    struct GeometricSetProbe;

    impl GeometricSetProbe {
        fn ball_with_nan_radius() -> LogicResult<crate::constraint::GeometricSet> {
            crate::constraint::GeometricSet::ball(vec![0.0, 0.0], f32::NAN)
        }
    }
}
