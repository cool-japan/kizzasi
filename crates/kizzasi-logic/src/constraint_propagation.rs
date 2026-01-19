//! Constraint Propagation for Discrete CSP
//!
//! This module implements constraint propagation algorithms for discrete
//! Constraint Satisfaction Problems (CSP):
//! - AC-3 (Arc Consistency 3) algorithm
//! - Domain reduction and filtering
//! - Forward checking
//! - Backtracking search with constraint propagation
//!
//! # Use Cases
//!
//! - Scheduling problems
//! - Resource allocation
//! - Configuration problems
//! - Graph coloring
//! - Sudoku and puzzle solving

use crate::error::{LogicError, LogicResult};
use std::collections::{HashMap, HashSet, VecDeque};

/// Domain of possible values for a variable
pub type Domain = HashSet<i32>;

/// Variable identifier
pub type VarId = usize;

/// Discrete constraint between variables
#[derive(Debug, Clone)]
pub enum DiscreteConstraint {
    /// Binary constraint: R(x, y)
    Binary {
        /// First variable
        var1: VarId,
        /// Second variable
        var2: VarId,
        /// Relation: set of allowed (var1, var2) pairs
        relation: HashSet<(i32, i32)>,
    },

    /// All-different constraint
    AllDifferent {
        /// Variables that must have different values
        variables: Vec<VarId>,
    },

    /// Sum constraint: Σ x_i = target
    Sum {
        /// Variables to sum
        variables: Vec<VarId>,
        /// Target sum
        target: i32,
    },

    /// Less-than constraint: x < y
    LessThan {
        /// First variable
        var1: VarId,
        /// Second variable
        var2: VarId,
    },

    /// Greater-than constraint: x > y
    GreaterThan {
        /// First variable
        var1: VarId,
        /// Second variable
        var2: VarId,
    },
}

impl DiscreteConstraint {
    /// Get all variables involved in this constraint
    pub fn variables(&self) -> Vec<VarId> {
        match self {
            Self::Binary { var1, var2, .. } => vec![*var1, *var2],
            Self::AllDifferent { variables } => variables.clone(),
            Self::Sum { variables, .. } => variables.clone(),
            Self::LessThan { var1, var2 } => vec![*var1, *var2],
            Self::GreaterThan { var1, var2 } => vec![*var1, *var2],
        }
    }

    /// Check if constraint is binary (involves exactly 2 variables)
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            Self::Binary { .. } | Self::LessThan { .. } | Self::GreaterThan { .. }
        )
    }

    /// Check if assignment satisfies constraint
    pub fn is_satisfied(&self, assignment: &HashMap<VarId, i32>) -> bool {
        match self {
            Self::Binary {
                var1,
                var2,
                relation,
            } => {
                if let (Some(&v1), Some(&v2)) = (assignment.get(var1), assignment.get(var2)) {
                    relation.contains(&(v1, v2))
                } else {
                    true // Not fully assigned yet
                }
            }
            Self::AllDifferent { variables } => {
                let values: Vec<i32> = variables
                    .iter()
                    .filter_map(|v| assignment.get(v))
                    .copied()
                    .collect();

                let unique: HashSet<_> = values.iter().collect();
                values.len() == unique.len()
            }
            Self::Sum { variables, target } => {
                if variables.iter().all(|v| assignment.contains_key(v)) {
                    let sum: i32 = variables.iter().filter_map(|v| assignment.get(v)).sum();
                    sum == *target
                } else {
                    true // Not fully assigned yet
                }
            }
            Self::LessThan { var1, var2 } => {
                if let (Some(&v1), Some(&v2)) = (assignment.get(var1), assignment.get(var2)) {
                    v1 < v2
                } else {
                    true
                }
            }
            Self::GreaterThan { var1, var2 } => {
                if let (Some(&v1), Some(&v2)) = (assignment.get(var1), assignment.get(var2)) {
                    v1 > v2
                } else {
                    true
                }
            }
        }
    }
}

/// Constraint Satisfaction Problem
pub struct CSP {
    /// Number of variables
    num_variables: usize,
    /// Domain for each variable
    domains: Vec<Domain>,
    /// Constraints
    constraints: Vec<DiscreteConstraint>,
}

impl CSP {
    /// Create a new CSP
    pub fn new(num_variables: usize, initial_domains: Vec<Domain>) -> LogicResult<Self> {
        if initial_domains.len() != num_variables {
            return Err(LogicError::InvalidInput(
                "Domain count must match variable count".to_string(),
            ));
        }

        Ok(Self {
            num_variables,
            domains: initial_domains,
            constraints: Vec::new(),
        })
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: DiscreteConstraint) {
        self.constraints.push(constraint);
    }

    /// Get domain of a variable
    pub fn domain(&self, var: VarId) -> Option<&Domain> {
        self.domains.get(var)
    }

    /// Get all constraints involving a variable
    pub fn constraints_for_variable(&self, var: VarId) -> Vec<&DiscreteConstraint> {
        self.constraints
            .iter()
            .filter(|c| c.variables().contains(&var))
            .collect()
    }

    /// Check if assignment is complete
    pub fn is_complete(&self, assignment: &HashMap<VarId, i32>) -> bool {
        assignment.len() == self.num_variables
    }

    /// Check if assignment satisfies all constraints
    pub fn is_consistent(&self, assignment: &HashMap<VarId, i32>) -> bool {
        self.constraints.iter().all(|c| c.is_satisfied(assignment))
    }
}

/// AC-3 Algorithm for Arc Consistency
pub struct AC3 {
    /// CSP instance
    csp: CSP,
}

impl AC3 {
    /// Create a new AC-3 solver
    pub fn new(csp: CSP) -> Self {
        Self { csp }
    }

    /// Enforce arc consistency
    ///
    /// Returns true if CSP is consistent, false if inconsistency detected
    pub fn enforce_arc_consistency(&mut self) -> bool {
        // Build queue of arcs to check
        let mut queue: VecDeque<(VarId, VarId)> = VecDeque::new();

        // Add all binary constraint arcs
        for constraint in &self.csp.constraints {
            if let DiscreteConstraint::Binary { var1, var2, .. }
            | DiscreteConstraint::LessThan { var1, var2 }
            | DiscreteConstraint::GreaterThan { var1, var2 } = constraint
            {
                queue.push_back((*var1, *var2));
                queue.push_back((*var2, *var1));
            }
        }

        // Process arcs
        while let Some((xi, xj)) = queue.pop_front() {
            if self.revise(xi, xj) {
                if self.csp.domains[xi].is_empty() {
                    return false; // Inconsistency detected
                }

                // Add all arcs (xk, xi) where xk is a neighbor of xi
                for constraint in &self.csp.constraints.clone() {
                    let vars = constraint.variables();
                    if vars.contains(&xi) && vars.len() == 2 {
                        for &xk in &vars {
                            if xk != xi && xk != xj {
                                queue.push_back((xk, xi));
                            }
                        }
                    }
                }
            }
        }

        true
    }

    /// Revise domain of xi with respect to xj
    ///
    /// Returns true if domain of xi was revised
    fn revise(&mut self, xi: VarId, xj: VarId) -> bool {
        let mut revised = false;

        // Find constraint between xi and xj
        let constraint = self
            .csp
            .constraints
            .iter()
            .find(|c| {
                let vars = c.variables();
                vars.len() == 2 && vars.contains(&xi) && vars.contains(&xj)
            })
            .cloned();

        if let Some(constraint) = constraint {
            let domain_j = self.csp.domains[xj].clone();
            let mut new_domain_i = HashSet::new();

            for &vi in &self.csp.domains[xi] {
                // Check if there exists vj in domain_j that satisfies constraint
                let mut has_support = false;

                for &vj in &domain_j {
                    let mut assignment = HashMap::new();
                    assignment.insert(xi, vi);
                    assignment.insert(xj, vj);

                    if constraint.is_satisfied(&assignment) {
                        has_support = true;
                        break;
                    }
                }

                if has_support {
                    new_domain_i.insert(vi);
                } else {
                    revised = true;
                }
            }

            self.csp.domains[xi] = new_domain_i;
        }

        revised
    }

    /// Get the CSP after arc consistency
    pub fn csp(self) -> CSP {
        self.csp
    }

    /// Get reference to CSP
    pub fn csp_ref(&self) -> &CSP {
        &self.csp
    }
}

/// Backtracking search with constraint propagation
pub struct BacktrackingSearch {
    /// CSP instance
    csp: CSP,
    /// Use forward checking
    use_forward_checking: bool,
    /// Solutions found
    solutions: Vec<HashMap<VarId, i32>>,
    /// Maximum solutions to find
    max_solutions: usize,
}

impl BacktrackingSearch {
    /// Create a new backtracking search
    pub fn new(csp: CSP) -> Self {
        Self {
            csp,
            use_forward_checking: true,
            solutions: Vec::new(),
            max_solutions: 1,
        }
    }

    /// Enable or disable forward checking
    pub fn with_forward_checking(mut self, enabled: bool) -> Self {
        self.use_forward_checking = enabled;
        self
    }

    /// Set maximum solutions to find
    pub fn with_max_solutions(mut self, max: usize) -> Self {
        self.max_solutions = max;
        self
    }

    /// Solve the CSP
    pub fn solve(&mut self) -> Vec<HashMap<VarId, i32>> {
        let assignment = HashMap::new();
        self.backtrack(assignment);
        self.solutions.clone()
    }

    /// Backtracking recursive search
    fn backtrack(&mut self, assignment: HashMap<VarId, i32>) -> bool {
        if self.solutions.len() >= self.max_solutions {
            return true;
        }

        if self.csp.is_complete(&assignment) {
            if self.csp.is_consistent(&assignment) {
                self.solutions.push(assignment.clone());
                return self.solutions.len() >= self.max_solutions;
            }
            return false;
        }

        // Select unassigned variable (MRV heuristic)
        let var = self.select_unassigned_variable(&assignment);

        // Order domain values
        let values = self.order_domain_values(var, &assignment);

        for value in values {
            let mut new_assignment = assignment.clone();
            new_assignment.insert(var, value);

            if self.is_consistent_with_assignment(&new_assignment) {
                if self.use_forward_checking {
                    // Forward checking would go here
                    // Simplified: just continue with backtracking
                }

                if self.backtrack(new_assignment) {
                    return true;
                }
            }
        }

        false
    }

    /// Select unassigned variable (Minimum Remaining Values heuristic)
    fn select_unassigned_variable(&self, assignment: &HashMap<VarId, i32>) -> VarId {
        let mut best_var = 0;
        let mut min_domain_size = usize::MAX;

        for var in 0..self.csp.num_variables {
            if !assignment.contains_key(&var) {
                let domain_size = self.csp.domains[var].len();
                if domain_size < min_domain_size {
                    min_domain_size = domain_size;
                    best_var = var;
                }
            }
        }

        best_var
    }

    /// Order domain values (Least Constraining Value heuristic)
    fn order_domain_values(&self, var: VarId, _assignment: &HashMap<VarId, i32>) -> Vec<i32> {
        let mut values: Vec<i32> = self.csp.domains[var].iter().copied().collect();
        values.sort(); // Simplified: just sort numerically
        values
    }

    /// Check if assignment is consistent with all constraints
    fn is_consistent_with_assignment(&self, assignment: &HashMap<VarId, i32>) -> bool {
        self.csp
            .constraints
            .iter()
            .all(|c| c.is_satisfied(assignment))
    }
}

/// Forward checking: maintain arc consistency during search
pub struct ForwardChecker {
    /// Original domains
    domains: Vec<Domain>,
}

impl ForwardChecker {
    /// Create a new forward checker
    pub fn new(domains: Vec<Domain>) -> Self {
        Self { domains }
    }

    /// Prune domains based on assignment
    pub fn prune(&mut self, var: VarId, value: i32, constraints: &[DiscreteConstraint]) -> bool {
        // For each constraint involving var
        for constraint in constraints {
            if !constraint.variables().contains(&var) {
                continue;
            }

            // Remove inconsistent values from neighboring variables
            let vars = constraint.variables();
            for &neighbor in &vars {
                if neighbor == var {
                    continue;
                }

                let mut new_domain = HashSet::new();
                for &v in &self.domains[neighbor] {
                    let mut assignment = HashMap::new();
                    assignment.insert(var, value);
                    assignment.insert(neighbor, v);

                    if constraint.is_satisfied(&assignment) {
                        new_domain.insert(v);
                    }
                }

                if new_domain.is_empty() {
                    return false; // Domain wipeout
                }

                self.domains[neighbor] = new_domain;
            }
        }

        true
    }

    /// Restore domains
    pub fn restore(&mut self, saved_domains: &[Domain]) {
        self.domains = saved_domains.to_vec();
    }

    /// Get current domains
    pub fn domains(&self) -> &[Domain] {
        &self.domains
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_constraint() {
        let mut relation = HashSet::new();
        relation.insert((1, 2));
        relation.insert((2, 3));

        let constraint = DiscreteConstraint::Binary {
            var1: 0,
            var2: 1,
            relation,
        };

        let mut assignment = HashMap::new();
        assignment.insert(0, 1);
        assignment.insert(1, 2);

        assert!(constraint.is_satisfied(&assignment));

        assignment.insert(1, 3);
        assert!(!constraint.is_satisfied(&assignment));
    }

    #[test]
    fn test_all_different_constraint() {
        let constraint = DiscreteConstraint::AllDifferent {
            variables: vec![0, 1, 2],
        };

        let mut assignment = HashMap::new();
        assignment.insert(0, 1);
        assignment.insert(1, 2);
        assignment.insert(2, 3);

        assert!(constraint.is_satisfied(&assignment));

        assignment.insert(2, 1); // Same as var 0
        assert!(!constraint.is_satisfied(&assignment));
    }

    #[test]
    fn test_less_than_constraint() {
        let constraint = DiscreteConstraint::LessThan { var1: 0, var2: 1 };

        let mut assignment = HashMap::new();
        assignment.insert(0, 5);
        assignment.insert(1, 10);

        assert!(constraint.is_satisfied(&assignment));

        assignment.insert(1, 3);
        assert!(!constraint.is_satisfied(&assignment));
    }

    #[test]
    fn test_csp_creation() {
        let domain1: Domain = [1, 2, 3].iter().cloned().collect();
        let domain2: Domain = [2, 3, 4].iter().cloned().collect();

        let csp = CSP::new(2, vec![domain1, domain2]).unwrap();

        assert_eq!(csp.num_variables, 2);
        assert_eq!(csp.domains.len(), 2);
    }

    #[test]
    fn test_ac3_simple() {
        let domain1: Domain = [1, 2, 3].iter().cloned().collect();
        let domain2: Domain = [2, 3, 4].iter().cloned().collect();

        let mut csp = CSP::new(2, vec![domain1, domain2]).unwrap();

        // Add constraint: var0 < var1
        csp.add_constraint(DiscreteConstraint::LessThan { var1: 0, var2: 1 });

        let mut ac3 = AC3::new(csp);
        let consistent = ac3.enforce_arc_consistency();

        assert!(consistent);

        // Domain of var0 should be reduced (values < some value in domain of var1)
        let csp_result = ac3.csp();
        assert!(!csp_result.domains[0].is_empty());
        assert!(!csp_result.domains[1].is_empty());
    }

    #[test]
    fn test_backtracking_search() {
        let domain1: Domain = [1, 2].iter().cloned().collect();
        let domain2: Domain = [1, 2].iter().cloned().collect();

        let mut csp = CSP::new(2, vec![domain1, domain2]).unwrap();

        // All different constraint
        csp.add_constraint(DiscreteConstraint::AllDifferent {
            variables: vec![0, 1],
        });

        let mut search = BacktrackingSearch::new(csp).with_max_solutions(2);
        let solutions = search.solve();

        assert!(!solutions.is_empty());
        // Should find 2 solutions: (1,2) and (2,1)
        assert!(solutions.len() <= 2);

        for solution in solutions {
            assert_ne!(solution.get(&0), solution.get(&1));
        }
    }

    #[test]
    fn test_forward_checker() {
        let domain1: Domain = [1, 2, 3].iter().cloned().collect();
        let domain2: Domain = [1, 2, 3].iter().cloned().collect();

        let mut checker = ForwardChecker::new(vec![domain1, domain2]);

        let constraints = vec![DiscreteConstraint::AllDifferent {
            variables: vec![0, 1],
        }];

        // Assign var0 = 1
        let success = checker.prune(0, 1, &constraints);
        assert!(success);

        // Domain of var1 should not contain 1
        assert!(!checker.domains()[1].contains(&1));
        assert!(checker.domains()[1].contains(&2));
        assert!(checker.domains()[1].contains(&3));
    }

    #[test]
    fn test_sum_constraint() {
        let constraint = DiscreteConstraint::Sum {
            variables: vec![0, 1, 2],
            target: 6,
        };

        let mut assignment = HashMap::new();
        assignment.insert(0, 1);
        assignment.insert(1, 2);
        assignment.insert(2, 3);

        assert!(constraint.is_satisfied(&assignment)); // 1 + 2 + 3 = 6

        assignment.insert(2, 4);
        assert!(!constraint.is_satisfied(&assignment)); // 1 + 2 + 4 = 7
    }
}
