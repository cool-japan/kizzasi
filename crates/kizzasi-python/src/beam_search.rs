//! Beam-search and rejection-sampling Python wrappers.
//!
//! Exposes three classes:
//!
//! - [`PyBeamSearch`] — plain beam search that maintains `beam_width` candidate
//!   sequences across autoregressive expansion steps.
//! - [`PyConstrainedBeamSearch`] — beam search extended with arbitrary Python
//!   callable constraints (either hard or soft with a configurable penalty).
//! - [`PyRejectionSampler`] — rejection sampler that tries up to `max_attempts`
//!   times to draw a sample satisfying all Python callable constraints before
//!   falling back to a configurable fallback strategy.
//!
//! ## Python-callable constraints
//!
//! All three classes accept Python callables as constraint functions.  The
//! callable receives a `list[float]` (the current candidate sequence) and must
//! return a `bool`.  Returning `True` means the constraint is satisfied;
//! returning `False` (or raising an exception) means it is violated.
//!
//! ## Builder-pattern note
//!
//! [`ConstrainedBeamSearch`] and [`RejectionSampler`] use a *consuming* builder
//! pattern on the Rust side (`add_constraint`, `max_attempts`, etc. all consume
//! `self` and return a new `Self`).  To accommodate this from PyO3 — where we
//! hold a `&mut self` reference — we store the inner value inside an
//! `Option<T>`.  Each mutating call does `self.inner.take()`, applies the
//! builder method, and stores the result back.  If `inner` is `None` (which
//! should never happen in normal usage), a `RuntimeError` is raised.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};

use kizzasi_inference::{
    BeamSearch, ConstrainedBeamSearch, ConstraintFn, FallbackStrategy, RejectionSampler,
    SamplingConfig,
};

use crate::predictor::to_py_err;
use crate::sampling::PySamplingConfig;

// ============================================================================
// Helper — wrap a Python callable into a ConstraintFn Arc
// ============================================================================

/// Wrap a Python callable object into a `ConstraintFn`.
///
/// The callable receives a `Vec<f32>` exposed to Python as a plain Python list
/// (converted from `&[f32]`).  Any Python-side exception or type error causes
/// the constraint to return `false` so the beam is treated as violating.
fn python_constraint_fn(callback: Py<PyAny>) -> ConstraintFn {
    Arc::new(move |sequence: &[f32]| -> bool {
        Python::attach(|py| -> bool {
            let py_seq: Vec<f32> = sequence.to_vec();
            callback
                .call1(py, (py_seq,))
                .and_then(|val| val.extract::<bool>(py))
                .unwrap_or(false)
        })
    })
}

// ============================================================================
// Helper — convert a slice of Beams to a Python list of dicts
// ============================================================================

/// Build a Python `list` of `dict` objects from a Rust `&[Beam]`.
///
/// Each dict has two keys:
///  - `"sequence"` → `numpy.ndarray` of shape `(n,)` dtype `float32`
///  - `"log_prob"` → Python `float`
fn beams_to_py_list(py: Python<'_>, beams: &[kizzasi_inference::Beam]) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for beam in beams {
        let d = PyDict::new(py);
        let seq_array: Array1<f32> = Array1::from_vec(beam.sequence.clone());
        d.set_item("sequence", seq_array.to_pyarray(py))?;
        d.set_item("log_prob", beam.log_prob)?;
        list.append(d)?;
    }
    Ok(list.into_any().unbind())
}

// ============================================================================
// PyBeamSearch
// ============================================================================

/// Plain beam search over a vocabulary.
///
/// Each call to :meth:`expand` takes a logits matrix of shape
/// ``(current_beams, vocab_size)`` (float32), expands every current beam into
/// ``beam_width`` candidates and keeps the top ``beam_width`` by average log
/// probability.
///
/// On construction there is exactly **one** empty beam; after the first
/// :meth:`expand` call there will be ``beam_width`` beams (or fewer if the
/// vocabulary is smaller than ``beam_width``).
///
/// Example
/// -------
/// .. code-block:: python
///
///     import numpy as np
///     import kizzasi
///
///     bs = kizzasi.BeamSearch(beam_width=3)
///     # first step: one beam -> logits shape (1, 8)
///     logits = np.random.randn(1, 8).astype(np.float32)
///     bs.expand(logits)
///     # subsequent steps: three beams -> logits shape (3, 8)
///     logits3 = np.random.randn(3, 8).astype(np.float32)
///     bs.expand(logits3)
///     print(bs.best_sequence())
#[pyclass(name = "BeamSearch")]
pub struct PyBeamSearch {
    inner: BeamSearch,
}

#[pymethods]
impl PyBeamSearch {
    /// Create a new beam search.
    ///
    /// Parameters
    /// ----------
    /// beam_width:
    ///     Number of candidate beams to maintain at each step.  Must be >= 1.
    #[new]
    pub fn new(beam_width: usize) -> PyResult<Self> {
        if beam_width == 0 {
            return Err(PyValueError::new_err("beam_width must be >= 1"));
        }
        Ok(Self {
            inner: BeamSearch::new(beam_width),
        })
    }

    /// Expand all current beams by one step.
    ///
    /// Parameters
    /// ----------
    /// logits:
    ///     Float32 array of shape ``(current_beams, vocab_size)``.
    ///     The number of rows **must** equal :meth:`num_beams`.
    pub fn expand<'py>(
        &mut self,
        py: Python<'py>,
        logits: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<()> {
        let arr: Array2<f32> = logits.as_array().to_owned();
        let _ = py;
        self.inner.expand(&arr).map_err(to_py_err)
    }

    /// Best candidate sequence so far, or ``None`` if there are no beams.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape ``(n,)`` float32, or ``None``.
    pub fn best_sequence<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.inner.best().map(|beam| {
            let arr: Array1<f32> = Array1::from_vec(beam.sequence.clone());
            arr.to_pyarray(py)
        })
    }

    /// Log probability of the best beam, or ``None`` if there are no beams.
    pub fn best_log_prob(&self) -> Option<f32> {
        self.inner.best().map(|beam| beam.log_prob)
    }

    /// Number of active beams.
    pub fn num_beams(&self) -> usize {
        self.inner.beams().len()
    }

    /// All active beams as a Python list of dicts.
    ///
    /// Each dict contains:
    ///   - ``"sequence"`` : numpy.ndarray of shape ``(n,)`` float32
    ///   - ``"log_prob"`` : float
    pub fn all_beams<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        beams_to_py_list(py, self.inner.beams())
    }

    fn __repr__(&self) -> String {
        format!(
            "BeamSearch(num_beams={}, best_log_prob={:?})",
            self.inner.beams().len(),
            self.inner.best().map(|b| b.log_prob),
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// PyConstrainedBeamSearch
// ============================================================================

/// Beam search extended with Python callable constraint functions.
///
/// Constraints can be **hard** (violating beams are filtered out) or **soft**
/// (violating beams have their log-probability reduced by a configurable
/// penalty).  By default constraints are hard.
///
/// Example
/// -------
/// .. code-block:: python
///
///     import numpy as np
///     import kizzasi
///
///     cbs = kizzasi.ConstrainedBeamSearch(beam_width=4)
///     # only accept sequences whose last value is less than 5.0
///     cbs.add_constraint(lambda seq: len(seq) == 0 or seq[-1] < 5.0)
///
///     logits = np.random.randn(1, 10).astype(np.float32)
///     cbs.expand(logits)
///     print(cbs.best_sequence())
#[pyclass(name = "ConstrainedBeamSearch")]
pub struct PyConstrainedBeamSearch {
    /// Wrapped `ConstrainedBeamSearch`.  Stored as `Option<_>` so we can use
    /// `take()` + re-assign to drive the consuming builder API on the Rust side.
    inner: Option<ConstrainedBeamSearch>,
    /// Cached beam width so we can surface it without digging into `inner`.
    beam_width: usize,
    /// Number of constraints that have been added so far.
    n_constraints: usize,
}

#[pymethods]
impl PyConstrainedBeamSearch {
    /// Create a new constrained beam search.
    ///
    /// Parameters
    /// ----------
    /// beam_width:
    ///     Number of candidate beams to maintain.  Must be >= 1.
    #[new]
    pub fn new(beam_width: usize) -> PyResult<Self> {
        if beam_width == 0 {
            return Err(PyValueError::new_err("beam_width must be >= 1"));
        }
        Ok(Self {
            inner: Some(ConstrainedBeamSearch::new(beam_width)),
            beam_width,
            n_constraints: 0,
        })
    }

    /// Register a hard constraint.
    ///
    /// Parameters
    /// ----------
    /// constraint_fn:
    ///     Python callable ``(sequence: list[float]) -> bool``.
    ///     ``True`` means the constraint is satisfied.
    pub fn add_constraint(&mut self, constraint_fn: Py<PyAny>) -> PyResult<()> {
        let rust_fn = python_constraint_fn(constraint_fn);
        let cbs = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "ConstrainedBeamSearch is in an invalid state",
            )
        })?;
        self.inner = Some(cbs.add_constraint(rust_fn));
        self.n_constraints += 1;
        Ok(())
    }

    /// Enable soft constraints with a configurable log-probability penalty.
    ///
    /// When soft constraints are enabled every beam that violates *any*
    /// constraint has ``penalty`` subtracted from its log-probability instead
    /// of being discarded.
    ///
    /// Parameters
    /// ----------
    /// penalty:
    ///     Non-negative log-probability penalty per violating beam.
    pub fn enable_soft_constraints(&mut self, penalty: f32) -> PyResult<()> {
        let cbs = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "ConstrainedBeamSearch is in an invalid state",
            )
        })?;
        self.inner = Some(cbs.with_soft_constraints(penalty));
        Ok(())
    }

    /// Expand all current beams by one step.
    ///
    /// Parameters
    /// ----------
    /// logits:
    ///     Float32 array of shape ``(current_beams, vocab_size)``.
    pub fn expand<'py>(
        &mut self,
        py: Python<'py>,
        logits: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<()> {
        let arr: Array2<f32> = logits.as_array().to_owned();
        let _ = py;
        self.inner
            .as_mut()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "ConstrainedBeamSearch is in an invalid state",
                )
            })?
            .expand(&arr)
            .map_err(to_py_err)
    }

    /// Best candidate sequence so far, or ``None`` if there are no beams.
    pub fn best_sequence<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.inner.as_ref().and_then(|cbs| {
            cbs.best().map(|beam| {
                let arr: Array1<f32> = Array1::from_vec(beam.sequence.clone());
                arr.to_pyarray(py)
            })
        })
    }

    /// Number of active beams.
    pub fn num_beams(&self) -> usize {
        self.inner
            .as_ref()
            .map(|cbs| cbs.beams().len())
            .unwrap_or(0)
    }

    /// Number of constraint functions that have been registered.
    pub fn num_constraints(&self) -> usize {
        self.n_constraints
    }

    /// All active beams as a Python list of dicts.
    ///
    /// Each dict contains:
    ///   - ``"sequence"`` : numpy.ndarray of shape ``(n,)`` float32
    ///   - ``"log_prob"`` : float
    pub fn all_beams<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            Some(cbs) => beams_to_py_list(py, cbs.beams()),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "ConstrainedBeamSearch is in an invalid state",
            )),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ConstrainedBeamSearch(beam_width={}, num_beams={}, num_constraints={})",
            self.beam_width,
            self.num_beams(),
            self.n_constraints,
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// PyRejectionSampler
// ============================================================================

/// Parse a fallback strategy name (case-insensitive).
fn parse_fallback_strategy(name: &str) -> PyResult<FallbackStrategy> {
    match name.to_lowercase().replace('-', "_").as_str() {
        "best_candidate" | "best" => Ok(FallbackStrategy::BestCandidate),
        "greedy" => Ok(FallbackStrategy::Greedy),
        "error" => Ok(FallbackStrategy::Error),
        other => Err(PyValueError::new_err(format!(
            "Unknown fallback strategy '{}'. Valid options: \
             best_candidate, greedy, error",
            other
        ))),
    }
}

/// Rejection sampler with Python callable constraint functions.
///
/// On each call to :meth:`sample` the sampler draws up to ``max_attempts``
/// candidates from the underlying base sampler and returns the first one that
/// satisfies all registered constraints.  If no valid candidate is found the
/// fallback strategy determines the return value.
///
/// Example
/// -------
/// .. code-block:: python
///
///     import numpy as np
///     import kizzasi
///
///     cfg = kizzasi.SamplingConfig()
///     cfg.strategy("temperature")
///     cfg.temperature(1.0)
///     cfg.seed(42)
///
///     rs = kizzasi.RejectionSampler(cfg)
///     # only accept indices < 3
///     rs.add_constraint(lambda seq: seq[-1] < 3.0)
///     rs.set_max_attempts(50)
///
///     logits = np.array([2.0, 2.5, 1.8, 0.1, 0.05], dtype=np.float32)
///     val = rs.sample(logits, context=[])
///     print(val)  # will be 0.0, 1.0, or 2.0
#[pyclass(name = "RejectionSampler")]
pub struct PyRejectionSampler {
    /// Wrapped `RejectionSampler`.  Stored as `Option<_>` for the same reason
    /// as in [`PyConstrainedBeamSearch`]: the Rust builder methods consume
    /// `self`, so we use `take()` + re-assignment.
    inner: Option<RejectionSampler>,
    /// Cached count of registered constraints.
    n_constraints: usize,
}

#[pymethods]
impl PyRejectionSampler {
    /// Construct a rejection sampler from a :class:`SamplingConfig`.
    ///
    /// The config is cloned into the sampler, so later mutations of the
    /// Python ``SamplingConfig`` object will not be reflected here.
    #[new]
    pub fn new(config: &PySamplingConfig) -> Self {
        let inner_config: SamplingConfig = config.inner.clone();
        Self {
            inner: Some(RejectionSampler::new(inner_config)),
            n_constraints: 0,
        }
    }

    /// Register a constraint function.
    ///
    /// Parameters
    /// ----------
    /// constraint_fn:
    ///     Python callable ``(sequence: list[float]) -> bool``.
    ///     Receives the *extended* context (context + current candidate).
    pub fn add_constraint(&mut self, constraint_fn: Py<PyAny>) -> PyResult<()> {
        let rust_fn = python_constraint_fn(constraint_fn);
        let rs = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("RejectionSampler is in an invalid state")
        })?;
        self.inner = Some(rs.add_constraint(rust_fn));
        self.n_constraints += 1;
        Ok(())
    }

    /// Set the maximum number of rejection attempts before the fallback
    /// strategy is triggered.
    pub fn set_max_attempts(&mut self, n: usize) -> PyResult<()> {
        let rs = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("RejectionSampler is in an invalid state")
        })?;
        self.inner = Some(rs.max_attempts(n));
        Ok(())
    }

    /// Configure the fallback strategy used when all attempts fail.
    ///
    /// Parameters
    /// ----------
    /// name:
    ///     One of ``"best_candidate"`` (or ``"best"``), ``"greedy"``,
    ///     ``"error"``.  Case-insensitive; hyphens and underscores are
    ///     interchangeable.
    pub fn set_fallback_strategy(&mut self, name: &str) -> PyResult<()> {
        let strategy = parse_fallback_strategy(name)?;
        let rs = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("RejectionSampler is in an invalid state")
        })?;
        self.inner = Some(rs.fallback_strategy(strategy));
        Ok(())
    }

    /// Draw a sample that satisfies all registered constraints.
    ///
    /// Parameters
    /// ----------
    /// logits:
    ///     1-D float32 NumPy array of vocabulary logits.
    /// context:
    ///     Current sequence context (list of floats).  The constraint
    ///     callable receives ``context + [candidate]``.
    ///
    /// Returns
    /// -------
    /// float
    ///     Sampled value (index cast to f32 for most strategies).
    pub fn sample<'py>(
        &mut self,
        py: Python<'py>,
        logits: PyReadonlyArray1<'py, f32>,
        context: Vec<f32>,
    ) -> PyResult<f32> {
        let arr: Array1<f32> = logits.as_array().to_owned();
        let _ = py;
        self.inner
            .as_mut()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("RejectionSampler is in an invalid state")
            })?
            .sample_with_rejection(&arr, &context)
            .map_err(to_py_err)
    }

    /// Number of constraint functions that have been registered.
    pub fn num_constraints(&self) -> usize {
        self.n_constraints
    }

    fn __repr__(&self) -> String {
        format!("RejectionSampler(num_constraints={})", self.n_constraints)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() {
        Python::initialize();
    }

    // 1. PyBeamSearch::new with valid beam_width succeeds
    #[test]
    fn test_beam_search_new_valid() {
        let bs = PyBeamSearch::new(3);
        assert!(bs.is_ok(), "expected Ok, got {:?}", bs.err());
    }

    // 2. PyBeamSearch::new(0) returns an error
    #[test]
    fn test_beam_search_new_zero_width_errors() {
        let bs = PyBeamSearch::new(0);
        assert!(bs.is_err(), "expected Err for beam_width=0");
    }

    // 3. A freshly created PyBeamSearch starts with exactly 1 beam
    #[test]
    fn test_beam_search_starts_with_one_beam() {
        let bs = PyBeamSearch::new(4).expect("beam search");
        assert_eq!(bs.num_beams(), 1, "should start with 1 empty beam");
    }

    // 4. PyConstrainedBeamSearch::new with valid width succeeds
    #[test]
    fn test_constrained_beam_search_new_valid() {
        let cbs = PyConstrainedBeamSearch::new(2);
        assert!(cbs.is_ok(), "expected Ok, got {:?}", cbs.err());
    }

    // 5. PyConstrainedBeamSearch::new(0) returns an error
    #[test]
    fn test_constrained_beam_search_new_zero_width_errors() {
        let cbs = PyConstrainedBeamSearch::new(0);
        assert!(cbs.is_err(), "expected Err for beam_width=0");
    }

    // 6. add_constraint increments n_constraints
    #[test]
    fn test_constrained_beam_search_add_constraint_increments_count() {
        setup();
        let mut cbs = PyConstrainedBeamSearch::new(2).expect("cbs");
        assert_eq!(cbs.num_constraints(), 0);

        Python::attach(|py| {
            let always_true: Py<PyAny> = py
                .eval(pyo3::ffi::c_str!("lambda seq: True"), None, None)
                .expect("lambda")
                .unbind();
            cbs.add_constraint(always_true).expect("add_constraint");
        });

        assert_eq!(cbs.num_constraints(), 1);

        Python::attach(|py| {
            let always_true2: Py<PyAny> = py
                .eval(pyo3::ffi::c_str!("lambda seq: True"), None, None)
                .expect("lambda2")
                .unbind();
            cbs.add_constraint(always_true2).expect("add_constraint 2");
        });

        assert_eq!(cbs.num_constraints(), 2);
    }

    // 7. enable_soft_constraints succeeds and leaves inner Some
    #[test]
    fn test_constrained_beam_search_enable_soft_constraints_ok() {
        let mut cbs = PyConstrainedBeamSearch::new(3).expect("cbs");
        let result = cbs.enable_soft_constraints(0.5_f32);
        assert!(result.is_ok(), "enable_soft_constraints should succeed");
        assert!(
            cbs.inner.is_some(),
            "inner should be Some after enable_soft_constraints"
        );
    }

    // 8. PyRejectionSampler::new, num_constraints starts at 0
    #[test]
    fn test_rejection_sampler_new_no_constraints() {
        let cfg = PySamplingConfig::new();
        let rs = PyRejectionSampler::new(&cfg);
        assert_eq!(rs.num_constraints(), 0);
        assert!(rs.inner.is_some());
    }

    // 9. set_max_attempts succeeds and leaves inner Some
    #[test]
    fn test_rejection_sampler_set_max_attempts_ok() {
        let cfg = PySamplingConfig::new();
        let mut rs = PyRejectionSampler::new(&cfg);
        let result = rs.set_max_attempts(200);
        assert!(result.is_ok(), "set_max_attempts should succeed");
        assert!(rs.inner.is_some());
    }

    // 10. set_fallback_strategy with all valid variants succeeds
    #[test]
    fn test_rejection_sampler_set_fallback_strategy_valid_variants() {
        let cfg = PySamplingConfig::new();
        for name in &[
            "best_candidate",
            "best",
            "greedy",
            "error",
            "GREEDY",
            "Best_Candidate",
        ] {
            let mut rs = PyRejectionSampler::new(&cfg);
            let result = rs.set_fallback_strategy(name);
            assert!(
                result.is_ok(),
                "set_fallback_strategy('{}') should succeed, got {:?}",
                name,
                result.err()
            );
        }
    }

    // 11. set_fallback_strategy with an invalid name returns Err
    #[test]
    fn test_rejection_sampler_set_fallback_strategy_invalid_returns_error() {
        let cfg = PySamplingConfig::new();
        let mut rs = PyRejectionSampler::new(&cfg);
        let result = rs.set_fallback_strategy("frobnicator");
        assert!(result.is_err(), "expected Err for unknown strategy name");
    }

    // 12. add_constraint increments n_constraints on PyRejectionSampler
    #[test]
    fn test_rejection_sampler_add_constraint_increments_count() {
        setup();
        let cfg = PySamplingConfig::new();
        let mut rs = PyRejectionSampler::new(&cfg);
        assert_eq!(rs.num_constraints(), 0);

        Python::attach(|py| {
            let always_true: Py<PyAny> = py
                .eval(pyo3::ffi::c_str!("lambda seq: True"), None, None)
                .expect("lambda")
                .unbind();
            rs.add_constraint(always_true).expect("add_constraint");
        });

        assert_eq!(rs.num_constraints(), 1);

        Python::attach(|py| {
            let always_false: Py<PyAny> = py
                .eval(pyo3::ffi::c_str!("lambda seq: False"), None, None)
                .expect("lambda2")
                .unbind();
            rs.add_constraint(always_false).expect("add_constraint 2");
        });

        assert_eq!(rs.num_constraints(), 2);
    }
}
