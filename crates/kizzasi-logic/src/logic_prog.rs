//! Datalog-Style Logic Programming Integration
//!
//! A lightweight Datalog-style engine for constraint reasoning over signal facts.
//! Supports bottom-up fixpoint evaluation, unification, and a bridge for asserting
//! signal constraint facts into the engine.
//!
//! # Example
//!
//! ```
//! use kizzasi_logic::logic_prog::{Atom, Term, Rule, DatalogEngine};
//!
//! let mut engine = DatalogEngine::new();
//! engine.add_fact(Atom { predicate: "parent".to_string(), args: vec![Term::sym("tom"), Term::sym("bob")] });
//! engine.add_rule(Rule::new(
//!     Atom { predicate: "ancestor".to_string(), args: vec![Term::var("X"), Term::var("Y")] },
//!     vec![Atom { predicate: "parent".to_string(), args: vec![Term::var("X"), Term::var("Y")] }],
//! ));
//! let derived = engine.evaluate().unwrap();
//! assert!(engine.entails(&derived, &Atom { predicate: "ancestor".to_string(), args: vec![Term::sym("tom"), Term::sym("bob")] }));
//! ```

use crate::error::{LogicError, LogicResult};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Term and Atom
// ============================================================================

/// A term in a logic atom: either a variable, a symbol, or an integer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    /// A logic variable (by convention, starts with a capital letter)
    Var(String),
    /// A ground symbol / string literal
    Sym(String),
    /// An integer constant
    Int(i64),
}

impl Term {
    /// Return `true` iff this term is ground (i.e., not a variable).
    pub fn is_ground(&self) -> bool {
        !matches!(self, Term::Var(_))
    }

    /// Construct a `Var` term.
    pub fn var(name: &str) -> Self {
        Term::Var(name.to_string())
    }

    /// Construct a `Sym` term.
    pub fn sym(name: &str) -> Self {
        Term::Sym(name.to_string())
    }

    /// Construct an `Int` term.
    pub fn int(v: i64) -> Self {
        Term::Int(v)
    }
}

/// A ground-or-partially-ground atom: a predicate applied to a list of terms.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Atom {
    /// Predicate name
    pub predicate: String,
    /// Argument list (may contain variables before substitution)
    pub args: Vec<Term>,
}

impl Atom {
    /// Return `true` iff all arguments are ground.
    pub fn is_ground(&self) -> bool {
        self.args.iter().all(Term::is_ground)
    }
}

// ============================================================================
// Rule
// ============================================================================

/// A Datalog rule: `head :- body[0], body[1], ...`
///
/// A rule with an empty body is a fact (always derivable).
#[derive(Debug, Clone)]
pub struct Rule {
    /// The atom to derive when the body is satisfied
    pub head: Atom,
    /// Body atoms that must all be satisfied (existential join)
    pub body: Vec<Atom>,
}

impl Rule {
    /// Create a rule with a non-empty body.
    pub fn new(head: Atom, body: Vec<Atom>) -> Self {
        Self { head, body }
    }

    /// Create a unit fact (rule with empty body).
    pub fn fact(head: Atom) -> Self {
        Self { head, body: vec![] }
    }
}

// ============================================================================
// Substitution type
// ============================================================================

/// A variable-to-term mapping produced by unification.
pub type Substitution = HashMap<String, Term>;

// ============================================================================
// Unification
// ============================================================================

/// Apply a substitution to a single term, recursively chasing variable bindings.
fn apply_term(term: &Term, subst: &Substitution) -> Term {
    match term {
        Term::Var(name) => {
            if let Some(bound) = subst.get(name) {
                apply_term(bound, subst)
            } else {
                term.clone()
            }
        }
        other => other.clone(),
    }
}

/// Apply a substitution to an atom, producing a new atom with all variables replaced.
pub fn apply_subst(atom: &Atom, subst: &Substitution) -> Atom {
    Atom {
        predicate: atom.predicate.clone(),
        args: atom.args.iter().map(|t| apply_term(t, subst)).collect(),
    }
}

/// Attempt to unify atom `a` (possibly with variables) against ground atom `b`,
/// extending `subst`. Returns `Some(new_subst)` on success, `None` on failure.
pub fn unify(a: &Atom, b: &Atom, subst: &Substitution) -> Option<Substitution> {
    if a.predicate != b.predicate || a.args.len() != b.args.len() {
        return None;
    }

    let mut s = subst.clone();

    for (ta, tb) in a.args.iter().zip(b.args.iter()) {
        let ta_applied = apply_term(ta, &s);
        let tb_applied = apply_term(tb, &s);

        match (&ta_applied, &tb_applied) {
            (Term::Var(name), other) => {
                // Bind variable to ground term
                s.insert(name.clone(), other.clone());
            }
            (other, Term::Var(name)) => {
                // Bind variable to ground term (symmetric)
                s.insert(name.clone(), other.clone());
            }
            (a_term, b_term) => {
                if a_term != b_term {
                    return None;
                }
            }
        }
    }

    Some(s)
}

// ============================================================================
// DatalogEngine
// ============================================================================

/// Bottom-up Datalog evaluation engine.
///
/// Facts are asserted directly; rules are applied iteratively until a fixpoint
/// is reached (or `max_iterations` is exceeded). All derived facts (including
/// base facts) are returned from [`DatalogEngine::evaluate`].
pub struct DatalogEngine {
    facts: HashSet<Atom>,
    rules: Vec<Rule>,
    max_iterations: usize,
    last_iterations: std::cell::Cell<usize>,
}

impl Default for DatalogEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DatalogEngine {
    /// Create a new engine with a default maximum of 1000 fixpoint iterations.
    pub fn new() -> Self {
        Self {
            facts: HashSet::new(),
            rules: Vec::new(),
            max_iterations: 1000,
            last_iterations: std::cell::Cell::new(0),
        }
    }

    /// Create an engine with a custom iteration limit.
    pub fn with_max_iterations(max_iter: usize) -> Self {
        Self {
            facts: HashSet::new(),
            rules: Vec::new(),
            max_iterations: max_iter,
            last_iterations: std::cell::Cell::new(0),
        }
    }

    /// Assert a base fact into the engine.
    pub fn add_fact(&mut self, atom: Atom) {
        self.facts.insert(atom);
    }

    /// Add a derivation rule to the engine.
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Run bottom-up fixpoint evaluation.
    ///
    /// Starts from the set of base facts and applies all rules until no new
    /// facts are derived (fixpoint) or `max_iterations` is exceeded.
    ///
    /// Returns the complete set of derived facts (including base facts).
    pub fn evaluate(&self) -> LogicResult<HashSet<Atom>> {
        let mut derived = self.facts.clone();
        let mut iteration = 0;

        loop {
            if iteration >= self.max_iterations {
                self.last_iterations.set(iteration);
                return Err(LogicError::InvalidInput(format!(
                    "DatalogEngine: fixpoint not reached after {} iterations",
                    self.max_iterations
                )));
            }

            let new_facts = self.derive_one_step(&derived);
            let added_any = new_facts.into_iter().fold(false, |acc, fact| {
                let is_new = !derived.contains(&fact);
                if is_new {
                    derived.insert(fact);
                }
                acc || is_new
            });

            iteration += 1;

            if !added_any {
                break;
            }
        }

        self.last_iterations.set(iteration);
        Ok(derived)
    }

    /// Derive all facts producible in one step from the current `known` set.
    fn derive_one_step(&self, known: &HashSet<Atom>) -> Vec<Atom> {
        let mut new_facts = Vec::new();

        for rule in &self.rules {
            // Generate all substitutions that satisfy the body
            let substs = self.join_body(&rule.body, known);
            for subst in substs {
                let head = apply_subst(&rule.head, &subst);
                if head.is_ground() {
                    new_facts.push(head);
                }
            }
        }

        new_facts
    }

    /// Find all substitutions that satisfy every atom in `body` against `known` facts.
    ///
    /// Uses a recursive nested-loop join (sound for Datalog's stratified semantics).
    fn join_body(&self, body: &[Atom], known: &HashSet<Atom>) -> Vec<Substitution> {
        if body.is_empty() {
            return vec![HashMap::new()];
        }

        let first = &body[0];
        let rest = &body[1..];

        let mut results = Vec::new();

        // For each known ground fact, try to unify with `first`
        for known_atom in known {
            let empty: Substitution = HashMap::new();
            if let Some(subst) = unify(first, known_atom, &empty) {
                // Recurse on remaining body with the extended substitution
                let rest_atoms: Vec<Atom> = rest.iter().map(|a| apply_subst(a, &subst)).collect();
                for mut child_subst in self.join_body(&rest_atoms, known) {
                    // Merge subst into child_subst
                    for (k, v) in &subst {
                        child_subst.entry(k.clone()).or_insert_with(|| v.clone());
                    }
                    results.push(child_subst);
                }
            }
        }

        results
    }

    /// Find all substitutions for which `apply_subst(query, subst)` is in `derived`.
    pub fn query(&self, derived: &HashSet<Atom>, query: &Atom) -> Vec<Substitution> {
        let mut results = Vec::new();

        for fact in derived {
            let empty: Substitution = HashMap::new();
            if let Some(subst) = unify(query, fact, &empty) {
                results.push(subst);
            }
        }

        results
    }

    /// Check whether a specific ground atom is in the derived set.
    pub fn entails(&self, derived: &HashSet<Atom>, atom: &Atom) -> bool {
        derived.contains(atom)
    }

    /// Return the number of fixpoint iterations taken during the last call to `evaluate`.
    pub fn last_iterations(&self) -> usize {
        self.last_iterations.get()
    }
}

// ============================================================================
// SignalFactBridge
// ============================================================================

/// A bridge that translates signal constraint checks into Datalog facts,
/// enabling logic-based violation reasoning over streaming data.
///
/// For each signal dimension and time step, one of two atoms is asserted:
/// - `in_range(dim_N, step_M, ok)` — value is within bounds
/// - `in_range(dim_N, step_M, violation)` — value is out of bounds
pub struct SignalFactBridge {
    engine: DatalogEngine,
    /// Current time step counter
    pub step: usize,
}

impl Default for SignalFactBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalFactBridge {
    /// Create a new bridge at step 0.
    pub fn new() -> Self {
        Self {
            engine: DatalogEngine::new(),
            step: 0,
        }
    }

    /// Assert a bounds fact for `dim` at the current step.
    ///
    /// Adds `in_range(dim_N, step_M, ok)` if `lo <= value <= hi`,
    /// otherwise `in_range(dim_N, step_M, violation)`.
    pub fn assert_bounds_fact(&mut self, dim: usize, value: f32, lo: f32, hi: f32) {
        let status = if value >= lo && value <= hi {
            "ok"
        } else {
            "violation"
        };

        let atom = Atom {
            predicate: "in_range".to_string(),
            args: vec![
                Term::sym(&format!("dim_{dim}")),
                Term::sym(&format!("step_{}", self.step)),
                Term::sym(status),
            ],
        };

        self.engine.add_fact(atom);
    }

    /// Assert an arbitrary custom fact into the engine.
    pub fn assert_custom_fact(&mut self, predicate: &str, args: Vec<Term>) {
        self.engine.add_fact(Atom {
            predicate: predicate.to_string(),
            args,
        });
    }

    /// Add a derivation rule to the underlying engine.
    pub fn add_constraint_rule(&mut self, rule: Rule) {
        self.engine.add_rule(rule);
    }

    /// Derive all facts and check whether any `violation` atom is entailed.
    ///
    /// A `violation` is signalled by the presence of any `in_range(_, _, violation)` fact,
    /// or any atom whose predicate is "violation".
    pub fn has_violations(&self) -> LogicResult<bool> {
        let derived = self.engine.evaluate()?;
        let has = derived.iter().any(|atom| {
            // Check for in_range(..., violation) facts
            if atom.predicate == "in_range" {
                if let Some(Term::Sym(status)) = atom.args.last() {
                    return status == "violation";
                }
            }
            // Check for standalone "violation" predicate
            atom.predicate == "violation"
        });
        Ok(has)
    }

    /// Return the list of dimension indices that have violations at any step.
    pub fn violated_dims(&self) -> LogicResult<Vec<usize>> {
        let derived = self.engine.evaluate()?;
        let mut dims = Vec::new();

        for atom in &derived {
            if atom.predicate == "in_range" && atom.args.len() == 3 {
                if let Term::Sym(status) = &atom.args[2] {
                    if status == "violation" {
                        if let Term::Sym(dim_str) = &atom.args[0] {
                            if let Some(rest) = dim_str.strip_prefix("dim_") {
                                if let Ok(dim) = rest.parse::<usize>() {
                                    if !dims.contains(&dim) {
                                        dims.push(dim);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        dims.sort();
        Ok(dims)
    }

    /// Advance to the next time step.
    pub fn advance_step(&mut self) {
        self.step += 1;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn atom(pred: &str, args: Vec<Term>) -> Atom {
        Atom {
            predicate: pred.to_string(),
            args,
        }
    }

    #[test]
    fn test_fact_entailment() {
        let mut engine = DatalogEngine::new();
        let fact = atom("foo", vec![Term::sym("a")]);
        engine.add_fact(fact.clone());
        let derived = engine.evaluate().expect("evaluate failed");
        assert!(
            engine.entails(&derived, &fact),
            "directly added fact should be entailed"
        );
    }

    #[test]
    fn test_rule_derivation_single_step() {
        let mut engine = DatalogEngine::new();

        // parent(tom, bob)
        engine.add_fact(atom("parent", vec![Term::sym("tom"), Term::sym("bob")]));

        // ancestor(X, Y) :- parent(X, Y)
        engine.add_rule(Rule::new(
            atom("ancestor", vec![Term::var("X"), Term::var("Y")]),
            vec![atom("parent", vec![Term::var("X"), Term::var("Y")])],
        ));

        let derived = engine.evaluate().expect("evaluate failed");

        let expected = atom("ancestor", vec![Term::sym("tom"), Term::sym("bob")]);
        assert!(
            engine.entails(&derived, &expected),
            "ancestor(tom, bob) should be derived from parent(tom, bob)"
        );
    }

    #[test]
    fn test_rule_derivation_chained() {
        let mut engine = DatalogEngine::new();

        // parent(a, b), parent(b, c)
        engine.add_fact(atom("parent", vec![Term::sym("a"), Term::sym("b")]));
        engine.add_fact(atom("parent", vec![Term::sym("b"), Term::sym("c")]));

        // ancestor(X, Y) :- parent(X, Y)
        engine.add_rule(Rule::new(
            atom("ancestor", vec![Term::var("X"), Term::var("Y")]),
            vec![atom("parent", vec![Term::var("X"), Term::var("Y")])],
        ));

        // ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
        engine.add_rule(Rule::new(
            atom("ancestor", vec![Term::var("X"), Term::var("Z")]),
            vec![
                atom("parent", vec![Term::var("X"), Term::var("Y")]),
                atom("ancestor", vec![Term::var("Y"), Term::var("Z")]),
            ],
        ));

        let derived = engine.evaluate().expect("evaluate failed");

        let expected = atom("ancestor", vec![Term::sym("a"), Term::sym("c")]);
        assert!(
            engine.entails(&derived, &expected),
            "ancestor(a, c) should be derivable via 2-hop chain"
        );
    }

    #[test]
    fn test_unification_vars() {
        // unify Atom("p", [Var("X"), Sym("b")]) with Atom("p", [Sym("a"), Sym("b")])
        let a = atom("p", vec![Term::var("X"), Term::sym("b")]);
        let b = atom("p", vec![Term::sym("a"), Term::sym("b")]);
        let empty: Substitution = HashMap::new();
        let result = unify(&a, &b, &empty);

        assert!(result.is_some(), "unification should succeed");
        let subst = result.unwrap();
        assert_eq!(
            subst.get("X"),
            Some(&Term::sym("a")),
            "X should be bound to Sym(\"a\")"
        );
    }

    #[test]
    fn test_query_returns_substitutions() {
        let mut engine = DatalogEngine::new();

        // parent(alice, bob), parent(alice, carol)
        engine.add_fact(atom("parent", vec![Term::sym("alice"), Term::sym("bob")]));
        engine.add_fact(atom("parent", vec![Term::sym("alice"), Term::sym("carol")]));

        let derived = engine.evaluate().expect("evaluate failed");

        // Query: parent(alice, Y)
        let query = atom("parent", vec![Term::sym("alice"), Term::var("Y")]);
        let substs = engine.query(&derived, &query);

        assert_eq!(
            substs.len(),
            2,
            "should find 2 substitutions for parent(alice, Y)"
        );
    }

    #[test]
    fn test_fixpoint_terminates() {
        // Cyclic rule: a(X) :- a(X)  — should terminate (no new facts added)
        let mut engine = DatalogEngine::with_max_iterations(10);
        engine.add_fact(atom("a", vec![Term::sym("x")]));
        engine.add_rule(Rule::new(
            atom("a", vec![Term::var("X")]),
            vec![atom("a", vec![Term::var("X")])],
        ));

        let result = engine.evaluate();
        // Should succeed (no new facts derived, fixpoint reached immediately)
        assert!(result.is_ok(), "cyclic rule should terminate: {:?}", result);
    }

    #[test]
    fn test_bridge_in_range() {
        let mut bridge = SignalFactBridge::new();
        bridge.assert_bounds_fact(0, 0.5, 0.0, 1.0);
        let has_viol = bridge.has_violations().expect("has_violations failed");
        assert!(!has_viol, "0.5 in [0, 1] should have no violations");
    }

    #[test]
    fn test_bridge_out_of_range() {
        let mut bridge = SignalFactBridge::new();
        bridge.assert_bounds_fact(0, 2.0, 0.0, 1.0);
        let has_viol = bridge.has_violations().expect("has_violations failed");
        assert!(has_viol, "2.0 not in [0, 1] should have a violation");
    }

    #[test]
    fn test_bridge_advance_step() {
        let mut bridge = SignalFactBridge::new();
        assert_eq!(bridge.step, 0, "step should start at 0");
        bridge.advance_step();
        assert_eq!(bridge.step, 1, "step should be 1 after advance_step()");
    }
}
