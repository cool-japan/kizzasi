//! Dense two-phase simplex solver for small linear programs
//!
//! This is a self-contained, pure-Rust LP solver used by the Benders
//! decomposition in [`crate::decomposition`]. It is deliberately scoped to the
//! canonical inequality form
//!
//! ```text
//! minimize    cᵀz
//! subject to  G z >= h
//!             z >= 0
//! ```
//!
//! and returns, besides the primal optimum, the **dual** vector
//! `π` of the `G z >= h` rows — Benders cuts are built from those multipliers.
//! When the program is infeasible the solver returns a Farkas certificate
//! (`π >= 0`, `Gᵀπ <= 0`, `πᵀh > 0`), which is exactly the extreme ray a
//! feasibility cut needs.
//!
//! Numerics: the tableau is carried in `f64` regardless of the `f32` interface
//! of the surrounding crate, and pivoting follows Bland's rule so the method
//! cannot cycle on degenerate vertices. Problem sizes here are small (tens of
//! rows), so the anti-cycling cost is irrelevant.

/// Absolute tolerance used for pivot selection and feasibility tests.
const EPS: f64 = 1e-9;

/// Outcome of a linear program solved by [`solve_ge_lp`].
#[derive(Debug, Clone)]
pub(crate) enum LpOutcome {
    /// A finite optimum was reached.
    Optimal {
        /// Optimal primal vector `z`.
        primal: Vec<f64>,
        /// Optimal dual vector `π` for the `G z >= h` rows.
        dual: Vec<f64>,
        /// Optimal objective value `cᵀz`.
        objective: f64,
    },
    /// The feasible region is empty; `farkas` is a certificate of that.
    Infeasible {
        /// Farkas certificate: `π >= 0`, `Gᵀπ <= 0`, `πᵀh > 0`.
        farkas: Vec<f64>,
    },
    /// The objective is unbounded below on the feasible region.
    Unbounded,
    /// The iteration budget was exhausted before a conclusion was reached.
    IterationLimit,
    /// The pivoting produced numerically inconsistent multipliers.
    NumericalFailure(String),
}

/// A simplex tableau in standard form `M z = b`, `z >= 0`.
struct Tableau {
    /// `rows x (cols + 1)` matrix; the last column holds the right-hand side.
    data: Vec<Vec<f64>>,
    /// Basic variable index per row.
    basis: Vec<usize>,
    /// Number of structural + slack + artificial columns.
    cols: usize,
    /// Number of constraint rows.
    rows: usize,
}

impl Tableau {
    fn value(&self, row: usize, col: usize) -> f64 {
        self.data
            .get(row)
            .and_then(|r| r.get(col))
            .copied()
            .unwrap_or(0.0)
    }

    fn rhs(&self, row: usize) -> f64 {
        self.value(row, self.cols)
    }

    /// Gauss-Jordan pivot on (`row`, `col`), making `col` the basis of `row`.
    fn pivot(&mut self, row: usize, col: usize) {
        let pivot_value = self.value(row, col);
        if pivot_value.abs() < EPS {
            return;
        }

        if let Some(pivot_row) = self.data.get_mut(row) {
            for entry in pivot_row.iter_mut() {
                *entry /= pivot_value;
            }
        }

        let pivot_row: Vec<f64> = match self.data.get(row) {
            Some(r) => r.clone(),
            None => return,
        };

        for (index, other) in self.data.iter_mut().enumerate() {
            if index == row {
                continue;
            }
            let factor = other.get(col).copied().unwrap_or(0.0);
            if factor.abs() < EPS {
                continue;
            }
            for (target, source) in other.iter_mut().zip(pivot_row.iter()) {
                *target -= factor * source;
            }
        }

        if let Some(slot) = self.basis.get_mut(row) {
            *slot = col;
        }
    }

    /// Current value of every column (0 for non-basic columns).
    fn solution(&self) -> Vec<f64> {
        let mut values = vec![0.0; self.cols];
        for row in 0..self.rows {
            if let Some(&basic) = self.basis.get(row) {
                if let Some(slot) = values.get_mut(basic) {
                    *slot = self.rhs(row);
                }
            }
        }
        values
    }

    /// Reduced costs `c_j - c_Bᵀ B⁻¹ M_j` and the objective value for `costs`.
    fn reduced_costs(&self, costs: &[f64]) -> (Vec<f64>, f64) {
        let mut reduced = costs.to_vec();
        reduced.resize(self.cols, 0.0);
        let mut objective = 0.0;

        for row in 0..self.rows {
            let basic = self.basis.get(row).copied().unwrap_or(0);
            let basic_cost = costs.get(basic).copied().unwrap_or(0.0);
            if basic_cost.abs() < EPS {
                continue;
            }
            for col in 0..self.cols {
                if let Some(slot) = reduced.get_mut(col) {
                    *slot -= basic_cost * self.value(row, col);
                }
            }
            objective += basic_cost * self.rhs(row);
        }

        (reduced, objective)
    }
}

/// Result of one simplex phase.
enum PhaseOutcome {
    Optimal,
    Unbounded,
    IterationLimit,
}

/// Run primal simplex on `tableau` for the given cost vector.
///
/// `forbidden` marks columns that must never enter the basis (the artificial
/// columns during phase II). Bland's rule (lowest eligible index) is used for
/// both the entering and the leaving choice, which guarantees termination.
fn run_simplex(
    tableau: &mut Tableau,
    costs: &[f64],
    forbidden: &[bool],
    max_iterations: usize,
) -> PhaseOutcome {
    for _ in 0..max_iterations {
        let (reduced, _) = tableau.reduced_costs(costs);

        // Bland's rule: smallest index with a strictly negative reduced cost.
        let entering = (0..tableau.cols).find(|&col| {
            !forbidden.get(col).copied().unwrap_or(false)
                && reduced.get(col).copied().unwrap_or(0.0) < -EPS
        });

        let Some(entering) = entering else {
            return PhaseOutcome::Optimal;
        };

        // Ratio test, ties broken by the smallest basic-variable index.
        let mut leaving: Option<usize> = None;
        let mut best_ratio = f64::INFINITY;
        for row in 0..tableau.rows {
            let coefficient = tableau.value(row, entering);
            if coefficient <= EPS {
                continue;
            }
            let ratio = tableau.rhs(row) / coefficient;
            let current_basic = tableau.basis.get(row).copied().unwrap_or(usize::MAX);
            let replace = match leaving {
                None => true,
                Some(best_row) => {
                    if ratio < best_ratio - EPS {
                        true
                    } else if ratio <= best_ratio + EPS {
                        let best_basic = tableau.basis.get(best_row).copied().unwrap_or(usize::MAX);
                        current_basic < best_basic
                    } else {
                        false
                    }
                }
            };
            if replace {
                leaving = Some(row);
                best_ratio = ratio;
            }
        }

        let Some(leaving) = leaving else {
            return PhaseOutcome::Unbounded;
        };

        tableau.pivot(leaving, entering);
    }

    PhaseOutcome::IterationLimit
}

/// Solve `min cᵀz s.t. G z >= h, z >= 0` with a two-phase primal simplex.
///
/// `g` is given row-major: `g[i]` is row `i` of `G` and must have `c.len()`
/// entries. Returns [`LpOutcome::NumericalFailure`] when the multipliers it
/// derives are not dual feasible — callers rely on that guarantee, so it is
/// checked rather than assumed.
pub(crate) fn solve_ge_lp(
    c: &[f64],
    g: &[Vec<f64>],
    h: &[f64],
    max_iterations: usize,
) -> LpOutcome {
    let n = c.len();
    let m = g.len();

    if h.len() != m || g.iter().any(|row| row.len() != n) {
        return LpOutcome::NumericalFailure(
            "constraint matrix and right-hand side shapes disagree".to_string(),
        );
    }

    // Trivial case: no constraints. The optimum is 0 unless some cost is
    // negative, in which case the program is unbounded on z >= 0.
    if m == 0 {
        if c.iter().any(|&cost| cost < -EPS) {
            return LpOutcome::Unbounded;
        }
        return LpOutcome::Optimal {
            primal: vec![0.0; n],
            dual: Vec::new(),
            objective: 0.0,
        };
    }

    // Columns: n structural, m surplus, m artificial.
    let surplus_offset = n;
    let artificial_offset = n + m;
    let cols = n + 2 * m;

    // Rows are normalised so every right-hand side is non-negative; `signs`
    // records the flip so the duals can be mapped back to the original rows.
    let mut signs = vec![1.0_f64; m];
    let mut data = Vec::with_capacity(m);
    for (i, row) in g.iter().enumerate() {
        let rhs = h.get(i).copied().unwrap_or(0.0);
        let sign = if rhs < 0.0 { -1.0 } else { 1.0 };
        if let Some(slot) = signs.get_mut(i) {
            *slot = sign;
        }

        let mut tableau_row = vec![0.0_f64; cols + 1];
        for (j, &coefficient) in row.iter().enumerate() {
            if let Some(slot) = tableau_row.get_mut(j) {
                *slot = sign * coefficient;
            }
        }
        if let Some(slot) = tableau_row.get_mut(surplus_offset + i) {
            *slot = -sign;
        }
        if let Some(slot) = tableau_row.get_mut(artificial_offset + i) {
            *slot = 1.0;
        }
        if let Some(slot) = tableau_row.get_mut(cols) {
            *slot = sign * rhs;
        }
        data.push(tableau_row);
    }

    let mut tableau = Tableau {
        data,
        basis: (0..m).map(|i| artificial_offset + i).collect(),
        cols,
        rows: m,
    };

    // ---------------------------------------------------------------- phase I
    let mut phase_one_costs = vec![0.0_f64; cols];
    for i in 0..m {
        if let Some(slot) = phase_one_costs.get_mut(artificial_offset + i) {
            *slot = 1.0;
        }
    }
    let no_forbidden = vec![false; cols];

    match run_simplex(
        &mut tableau,
        &phase_one_costs,
        &no_forbidden,
        max_iterations,
    ) {
        PhaseOutcome::Optimal => {}
        PhaseOutcome::Unbounded => {
            return LpOutcome::NumericalFailure(
                "phase I of the simplex reported an unbounded objective".to_string(),
            )
        }
        PhaseOutcome::IterationLimit => return LpOutcome::IterationLimit,
    }

    let (phase_one_reduced, phase_one_objective) = tableau.reduced_costs(&phase_one_costs);

    if phase_one_objective > 1e-7 {
        // Infeasible. The phase-I multipliers form a Farkas certificate:
        // w_i = 1 - reduced_cost(artificial_i), then undo the row flips.
        let farkas: Vec<f64> = (0..m)
            .map(|i| {
                let reduced = phase_one_reduced
                    .get(artificial_offset + i)
                    .copied()
                    .unwrap_or(0.0);
                signs.get(i).copied().unwrap_or(1.0) * (1.0 - reduced)
            })
            .collect();

        if let Err(reason) = check_farkas(&farkas, g, h) {
            return LpOutcome::NumericalFailure(reason);
        }
        return LpOutcome::Infeasible { farkas };
    }

    // Drive any artificial that is still basic (at level zero) out of the basis
    // so phase II cannot re-inflate it.
    for row in 0..m {
        let basic = tableau.basis.get(row).copied().unwrap_or(0);
        if basic < artificial_offset {
            continue;
        }
        let replacement = (0..artificial_offset).find(|&col| tableau.value(row, col).abs() > EPS);
        if let Some(col) = replacement {
            tableau.pivot(row, col);
        }
        // No replacement means the row is redundant: it is all zeros outside
        // the artificial columns, so later pivots cannot change it and the
        // artificial stays pinned at zero.
    }

    // --------------------------------------------------------------- phase II
    let mut phase_two_costs = vec![0.0_f64; cols];
    for (j, &cost) in c.iter().enumerate() {
        if let Some(slot) = phase_two_costs.get_mut(j) {
            *slot = cost;
        }
    }
    let mut forbidden = vec![false; cols];
    for i in 0..m {
        if let Some(slot) = forbidden.get_mut(artificial_offset + i) {
            *slot = true;
        }
    }

    match run_simplex(&mut tableau, &phase_two_costs, &forbidden, max_iterations) {
        PhaseOutcome::Optimal => {}
        PhaseOutcome::Unbounded => return LpOutcome::Unbounded,
        PhaseOutcome::IterationLimit => return LpOutcome::IterationLimit,
    }

    let (reduced, objective) = tableau.reduced_costs(&phase_two_costs);
    let values = tableau.solution();
    let primal: Vec<f64> = values.iter().take(n).copied().collect();

    // Duals: reduced_cost(artificial_i) = -w_i, then undo the row flips.
    let dual: Vec<f64> = (0..m)
        .map(|i| {
            let reduced_cost = reduced.get(artificial_offset + i).copied().unwrap_or(0.0);
            signs.get(i).copied().unwrap_or(1.0) * (-reduced_cost)
        })
        .collect();

    if let Err(reason) = check_dual_feasible(&dual, c, g) {
        return LpOutcome::NumericalFailure(reason);
    }

    LpOutcome::Optimal {
        primal,
        dual,
        objective,
    }
}

/// Verify that `dual` is feasible for `max hᵀπ s.t. Gᵀπ <= c, π >= 0`.
///
/// Benders optimality cuts are only valid for a dual-feasible multiplier
/// vector, so this is checked explicitly instead of being assumed from the
/// tableau bookkeeping.
fn check_dual_feasible(dual: &[f64], c: &[f64], g: &[Vec<f64>]) -> Result<(), String> {
    let tolerance = 1e-5 * (1.0 + c.iter().fold(0.0_f64, |acc, v| acc.max(v.abs())));

    for (i, &value) in dual.iter().enumerate() {
        if value < -tolerance {
            return Err(format!("dual multiplier {i} is negative ({value})"));
        }
    }

    for (j, &cost) in c.iter().enumerate() {
        let mut column = 0.0;
        for (i, row) in g.iter().enumerate() {
            column += dual.get(i).copied().unwrap_or(0.0) * row.get(j).copied().unwrap_or(0.0);
        }
        if column > cost + tolerance {
            return Err(format!(
                "dual infeasible in column {j}: (Gᵀπ)_j = {column} exceeds c_j = {cost}"
            ));
        }
    }

    Ok(())
}

/// Verify a Farkas certificate: `π >= 0`, `Gᵀπ <= 0`, `πᵀh > 0`.
fn check_farkas(farkas: &[f64], g: &[Vec<f64>], h: &[f64]) -> Result<(), String> {
    let scale = farkas.iter().fold(1.0_f64, |acc, v| acc.max(v.abs()));
    let tolerance = 1e-5 * scale;

    for (i, &value) in farkas.iter().enumerate() {
        if value < -tolerance {
            return Err(format!("Farkas multiplier {i} is negative ({value})"));
        }
    }

    let columns = g.first().map(|row| row.len()).unwrap_or(0);
    for j in 0..columns {
        let mut column = 0.0;
        for (i, row) in g.iter().enumerate() {
            column += farkas.get(i).copied().unwrap_or(0.0) * row.get(j).copied().unwrap_or(0.0);
        }
        if column > tolerance {
            return Err(format!(
                "Farkas certificate violates Gᵀπ <= 0 in column {j}: {column}"
            ));
        }
    }

    let product: f64 = farkas
        .iter()
        .zip(h.iter())
        .map(|(&pi, &rhs)| pi * rhs)
        .sum();
    if product <= tolerance {
        return Err(format!(
            "Farkas certificate does not separate the right-hand side (πᵀh = {product})"
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_covering_lp() {
        // min 2y  s.t.  y >= 1,  y >= 0  ->  y = 1, obj 2, dual 2
        let outcome = solve_ge_lp(&[2.0], &[vec![1.0]], &[1.0], 200);
        match outcome {
            LpOutcome::Optimal {
                primal,
                dual,
                objective,
            } => {
                assert!((primal[0] - 1.0).abs() < 1e-6, "primal {primal:?}");
                assert!((objective - 2.0).abs() < 1e-6, "objective {objective}");
                assert!((dual[0] - 2.0).abs() < 1e-6, "dual {dual:?}");
            }
            other => panic!("expected optimal, got {other:?}"),
        }
    }

    #[test]
    fn test_two_variable_lp_matches_known_optimum() {
        // min 3a + 5b
        // s.t.  a + b >= 4
        //       a      >= 1
        //              b >= 1
        // Optimum: a = 3, b = 1, objective 14.
        let g = vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let outcome = solve_ge_lp(&[3.0, 5.0], &g, &[4.0, 1.0, 1.0], 500);
        match outcome {
            LpOutcome::Optimal {
                primal, objective, ..
            } => {
                assert!((objective - 14.0).abs() < 1e-5, "objective {objective}");
                assert!((primal[0] - 3.0).abs() < 1e-5, "primal {primal:?}");
                assert!((primal[1] - 1.0).abs() < 1e-5, "primal {primal:?}");
            }
            other => panic!("expected optimal, got {other:?}"),
        }
    }

    #[test]
    fn test_strong_duality_holds() {
        // The dual objective hᵀπ must equal the primal objective.
        let g = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let h = [6.0, 9.0];
        let c = [4.0, 5.0];
        match solve_ge_lp(&c, &g, &h, 500) {
            LpOutcome::Optimal {
                dual, objective, ..
            } => {
                let dual_objective: f64 =
                    dual.iter().zip(h.iter()).map(|(&pi, &rhs)| pi * rhs).sum();
                assert!(
                    (dual_objective - objective).abs() < 1e-5,
                    "strong duality violated: primal {objective}, dual {dual_objective}"
                );
            }
            other => panic!("expected optimal, got {other:?}"),
        }
    }

    #[test]
    fn test_infeasible_lp_returns_farkas_certificate() {
        // -y >= 1 with y >= 0 has no solution.
        match solve_ge_lp(&[1.0], &[vec![-1.0]], &[1.0], 200) {
            LpOutcome::Infeasible { farkas } => {
                assert!(farkas[0] > 0.0, "certificate must be positive: {farkas:?}");
            }
            other => panic!("expected infeasible, got {other:?}"),
        }
    }

    #[test]
    fn test_unbounded_lp_detected() {
        // min -y  s.t. y >= 1 is unbounded below.
        match solve_ge_lp(&[-1.0], &[vec![1.0]], &[1.0], 200) {
            LpOutcome::Unbounded => {}
            other => panic!("expected unbounded, got {other:?}"),
        }
    }

    #[test]
    fn test_negative_rhs_row_is_normalised() {
        // min y  s.t. -y >= -5  (i.e. y <= 5), y >= 0 -> y = 0.
        match solve_ge_lp(&[1.0], &[vec![-1.0]], &[-5.0], 200) {
            LpOutcome::Optimal {
                primal, objective, ..
            } => {
                assert!(primal[0].abs() < 1e-6, "primal {primal:?}");
                assert!(objective.abs() < 1e-6);
            }
            other => panic!("expected optimal, got {other:?}"),
        }
    }

    #[test]
    fn test_degenerate_problem_terminates() {
        // Redundant duplicated rows create a degenerate vertex; Bland's rule
        // must still terminate at the optimum.
        let g = vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 0.0]];
        match solve_ge_lp(&[1.0, 1.0], &g, &[2.0, 2.0, 1.0], 500) {
            LpOutcome::Optimal { objective, .. } => {
                assert!((objective - 2.0).abs() < 1e-5, "objective {objective}");
            }
            other => panic!("expected optimal, got {other:?}"),
        }
    }
}
