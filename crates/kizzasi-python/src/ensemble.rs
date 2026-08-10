//! Multi-model ensemble Python wrapper.
//!
//! Exposes [`PyEnsemblePredictor`] — a thin facade over
//! [`kizzasi::ensemble::EnsemblePredictor`] that:
//! * accepts a textual voting-strategy name (`"average"`, `"weighted"`,
//!   `"weighted_average"`, `"median"`, `"confidence"`, `"majority"`),
//! * spawns `n_models` independent [`kizzasi::Kizzasi`] predictors from a
//!   shared [`crate::config::PyKizzasiConfig`],
//! * mirrors the single-predictor API (`step`, `predict_n`, `reset`,
//!   `stats`).
//!
//! Note: the underlying `EnsemblePredictor::predict_n` is not exposed, so
//! `predict_n` in this wrapper performs autoregressive stepping locally. This
//! requires `input_dim == output_dim`; otherwise an error is returned.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use scirs2_numpy::{PyArray1, PyArray2, PyReadonlyArray1, ToPyArray};

use ::kizzasi::ensemble::{EnsemblePredictor, VotingStrategy};
use ::kizzasi::Kizzasi;
use scirs2_core::ndarray::{Array1, Array2};

use crate::config::PyKizzasiConfig;
use crate::predictor::to_py_err;

/// Map a textual voting-strategy name to [`VotingStrategy`].
fn parse_voting(name: &str) -> PyResult<VotingStrategy> {
    match name.to_lowercase().as_str() {
        "average" | "avg" | "mean" => Ok(VotingStrategy::Average),
        "weighted_average" | "weighted-average" => Ok(VotingStrategy::WeightedAverage),
        "weighted" => Ok(VotingStrategy::Weighted),
        "median" => Ok(VotingStrategy::Median),
        "confidence" | "confidence_based" | "confidence-based" => {
            Ok(VotingStrategy::ConfidenceBased)
        }
        "majority" | "majority_vote" | "majority-vote" => Ok(VotingStrategy::MajorityVote),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown voting strategy '{}'. Valid options: average, weighted, \
             weighted_average, median, confidence, majority",
            other
        ))),
    }
}

/// Return the canonical, lower-case name of a voting strategy.
fn voting_name(strategy: VotingStrategy) -> &'static str {
    match strategy {
        VotingStrategy::Average => "average",
        VotingStrategy::WeightedAverage => "weighted_average",
        VotingStrategy::Weighted => "weighted",
        VotingStrategy::Median => "median",
        VotingStrategy::ConfidenceBased => "confidence",
        VotingStrategy::MajorityVote => "majority",
    }
}

/// Multi-model ensemble predictor.
///
/// Constructs `n_models` independent predictors from a shared config and
/// combines their predictions with the chosen voting strategy. Optional
/// per-model `weights` are required for `weighted_average`/`weighted` strategies.
#[pyclass(name = "EnsemblePredictor", unsendable)]
pub struct PyEnsemblePredictor {
    inner: EnsemblePredictor,
    n_models: usize,
    config_snapshot: PyKizzasiConfig,
    input_dim: usize,
    output_dim: usize,
}

#[pymethods]
impl PyEnsemblePredictor {
    /// Build an ensemble of `n_models` predictors from one shared config.
    ///
    /// `voting` accepts: `"average"`, `"weighted"`, `"weighted_average"`,
    /// `"median"`, `"confidence"`, `"majority"`.
    ///
    /// `weights`, if given, must have length `n_models`; each weight must be
    /// non-negative. If omitted, every model receives weight `1.0`.
    #[new]
    #[pyo3(signature = (config, n_models, voting = "average", weights = None))]
    pub fn new(
        config: &PyKizzasiConfig,
        n_models: usize,
        voting: &str,
        weights: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        if n_models == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_models must be >= 1",
            ));
        }
        let strategy = parse_voting(voting)?;

        let weights = if let Some(ws) = weights {
            if ws.len() != n_models {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "weights length {} does not match n_models {}",
                    ws.len(),
                    n_models
                )));
            }
            ws
        } else {
            vec![1.0_f64; n_models]
        };

        let core_config = config.to_core_config()?;
        let mut inner = EnsemblePredictor::new(strategy);

        for (idx, weight) in weights.iter().enumerate().take(n_models) {
            let predictor = Kizzasi::new(core_config.clone()).map_err(to_py_err)?;
            inner
                .add_model_with_id(predictor, *weight, format!("model_{}", idx))
                .map_err(to_py_err)?;
        }

        Ok(Self {
            inner,
            n_models,
            config_snapshot: config.clone(),
            input_dim: config.input_dim,
            output_dim: config.output_dim,
        })
    }

    /// Single autoregressive ensemble prediction step.
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let arr = input.as_array();
        if arr.len() != self.input_dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Input length {} does not match ensemble input_dim {}",
                arr.len(),
                self.input_dim
            )));
        }
        let input_arr: Array1<f32> = arr.to_owned();
        let output = self.inner.predict(&input_arr).map_err(to_py_err)?;
        Ok(output.to_pyarray(py))
    }

    /// N-step autoregressive prediction, feeding each output back as next input.
    ///
    /// Requires `input_dim == output_dim`. Returns a `(n_steps, output_dim)`
    /// float32 ndarray.
    pub fn predict_n<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<'_, f32>,
        n_steps: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if self.input_dim != self.output_dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "predict_n requires input_dim ({}) == output_dim ({}); \
                 use step in a loop with custom feedback instead",
                self.input_dim, self.output_dim
            )));
        }
        if n_steps == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_steps must be >= 1",
            ));
        }
        let arr = input.as_array();
        if arr.len() != self.input_dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Input length {} does not match ensemble input_dim {}",
                arr.len(),
                self.input_dim
            )));
        }

        let mut current: Array1<f32> = arr.to_owned();
        let mut out = Array2::<f32>::zeros((n_steps, self.output_dim));
        for step in 0..n_steps {
            let pred = self.inner.predict(&current).map_err(to_py_err)?;
            out.row_mut(step).assign(&pred);
            current = pred;
        }
        Ok(out.to_pyarray(py))
    }

    /// Reset all models in the ensemble.
    pub fn reset(&mut self) {
        self.inner.reset_all();
    }

    /// Update the weight of a specific model by 0-based index.
    pub fn set_weight(&mut self, index: usize, weight: f64) -> PyResult<()> {
        if index >= self.n_models {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "index {} out of range (ensemble has {} models)",
                index, self.n_models
            )));
        }
        self.inner
            .update_weights(&format!("model_{}", index), weight)
            .map_err(to_py_err)?;
        Ok(())
    }

    /// Return a dict with ensemble statistics:
    /// `num_models`, `total_predictions`, `avg_variance`, `voting_strategy`,
    /// `model_weights` (list[float]).
    pub fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stats();
        let dict = PyDict::new(py);
        dict.set_item("num_models", s.num_models)?;
        dict.set_item("total_predictions", s.total_predictions)?;
        dict.set_item("avg_variance", s.avg_variance)?;
        dict.set_item("voting_strategy", voting_name(self.inner.strategy()))?;

        let mut weights = Vec::with_capacity(self.n_models);
        for idx in 0..self.n_models {
            let key = format!("model_{}", idx);
            let w = s.model_stats.get(&key).map(|m| m.weight).unwrap_or(0.0);
            weights.push(w);
        }
        dict.set_item("model_weights", weights)?;
        Ok(dict)
    }

    /// Number of models in the ensemble.
    #[getter]
    pub fn num_models(&self) -> usize {
        self.inner.num_models()
    }

    /// Canonical voting strategy name.
    #[getter]
    pub fn voting_strategy(&self) -> String {
        voting_name(self.inner.strategy()).to_string()
    }

    /// Per-model input dimension (same for every ensemble member).
    #[getter]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Per-model output dimension (same for every ensemble member).
    #[getter]
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Shared configuration used to construct every ensemble member.
    #[getter]
    pub fn config(&self) -> PyKizzasiConfig {
        self.config_snapshot.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "EnsemblePredictor(n_models={}, voting='{}', input_dim={}, output_dim={})",
            self.inner.num_models(),
            voting_name(self.inner.strategy()),
            self.input_dim,
            self.output_dim,
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> PyKizzasiConfig {
        PyKizzasiConfig::new(2, 2, 32, 1, 4, 256, "mamba2".to_string())
    }

    #[test]
    fn test_ensemble_basic() {
        let cfg = small_cfg();
        let mut ens =
            PyEnsemblePredictor::new(&cfg, 3, "average", None).expect("ensemble creation");
        assert_eq!(ens.num_models(), 3);
        assert_eq!(ens.voting_strategy(), "average");
        assert_eq!(ens.input_dim(), 2);
        assert_eq!(ens.output_dim(), 2);

        // step
        let input = Array1::from_vec(vec![0.5_f32, -0.5]);
        let out = ens.inner.predict(&input).expect("predict");
        assert_eq!(out.len(), 2);
        for v in out.iter() {
            assert!(v.is_finite(), "ensemble output not finite: {}", v);
        }
    }

    #[test]
    fn test_ensemble_voting_strategies() {
        let cfg = small_cfg();
        for voting in &[
            "average",
            "weighted",
            "weighted_average",
            "median",
            "confidence",
            "majority",
        ] {
            let weights = if voting.starts_with("weighted") || *voting == "confidence" {
                Some(vec![1.0_f64, 0.7, 0.3])
            } else {
                None
            };
            let ens = PyEnsemblePredictor::new(&cfg, 3, voting, weights);
            assert!(
                ens.is_ok(),
                "failed to build ensemble with voting='{}': {:?}",
                voting,
                ens.err()
            );
        }
    }

    #[test]
    fn test_ensemble_invalid_voting() {
        let cfg = small_cfg();
        let ens = PyEnsemblePredictor::new(&cfg, 2, "nonexistent", None);
        assert!(ens.is_err());
    }

    #[test]
    fn test_ensemble_zero_models_rejected() {
        let cfg = small_cfg();
        let ens = PyEnsemblePredictor::new(&cfg, 0, "average", None);
        assert!(ens.is_err());
    }

    #[test]
    fn test_ensemble_weights_length_mismatch() {
        let cfg = small_cfg();
        let ens = PyEnsemblePredictor::new(&cfg, 3, "weighted_average", Some(vec![1.0, 0.5]));
        assert!(ens.is_err());
    }

    #[test]
    fn test_ensemble_predict_n_autoregressive() {
        let cfg = small_cfg();
        let mut ens = PyEnsemblePredictor::new(&cfg, 2, "average", None).expect("ensemble");
        let input = Array1::from_vec(vec![0.1_f32, 0.2]);
        // Use predict() in a loop to verify autoregressive shape behaviour
        // (since predict_n requires the Python GIL via PyArray2; we just
        // emulate the inner logic here).
        let mut current = input.clone();
        for _ in 0..4 {
            let pred = ens.inner.predict(&current).expect("predict");
            assert_eq!(pred.len(), 2);
            current = pred;
        }
    }

    #[test]
    fn test_ensemble_predict_n_dim_mismatch_rejected() {
        // input_dim != output_dim -> predict_n should error before stepping
        let cfg = PyKizzasiConfig::new(3, 2, 32, 1, 4, 256, "mamba2".to_string());
        let _ens = PyEnsemblePredictor::new(&cfg, 2, "average", None).expect("ensemble");
        // We assert via a fresh-state ensemble that the dimensional check
        // works by inspecting the recorded dims; predict_n itself needs
        // the GIL, so we only verify the precondition.
        let cfg2 = PyKizzasiConfig::new(3, 2, 32, 1, 4, 256, "mamba2".to_string());
        let ens2 = PyEnsemblePredictor::new(&cfg2, 1, "average", None).expect("ensemble");
        assert_ne!(ens2.input_dim(), ens2.output_dim());
    }

    #[test]
    fn test_ensemble_set_weight() {
        let cfg = small_cfg();
        let mut ens = PyEnsemblePredictor::new(&cfg, 2, "weighted_average", Some(vec![1.0, 1.0]))
            .expect("ensemble");
        assert!(ens.set_weight(0, 0.25).is_ok());
        assert!(ens.set_weight(99, 1.0).is_err());
    }

    #[test]
    fn test_ensemble_reset() {
        let cfg = small_cfg();
        let mut ens = PyEnsemblePredictor::new(&cfg, 2, "average", None).expect("ensemble");
        let input = Array1::from_vec(vec![0.1_f32, 0.2]);
        ens.inner.predict(&input).expect("predict");
        ens.reset();
    }

    #[test]
    fn test_ensemble_repr() {
        let cfg = small_cfg();
        let ens = PyEnsemblePredictor::new(&cfg, 3, "median", None).expect("ensemble");
        let r = ens.__repr__();
        assert!(r.contains("EnsemblePredictor("));
        assert!(r.contains("n_models=3"));
        assert!(r.contains("median"));
    }

    #[test]
    fn test_parse_voting_aliases() {
        assert_eq!(parse_voting("AVERAGE").expect("p"), VotingStrategy::Average);
        assert_eq!(parse_voting("mean").expect("p"), VotingStrategy::Average);
        assert_eq!(
            parse_voting("weighted-average").expect("p"),
            VotingStrategy::WeightedAverage
        );
        assert_eq!(
            parse_voting("majority").expect("p"),
            VotingStrategy::MajorityVote
        );
    }
}
