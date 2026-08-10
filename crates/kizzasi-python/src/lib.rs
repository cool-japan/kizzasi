//! # kizzasi-python
//!
//! PyO3-based Python bindings for the Kizzasi AGSP (Autoregressive General-Purpose
//! Signal Predictor) ecosystem.
//!
//! ## Usage from Python
//!
//! ```python
//! import kizzasi
//!
//! # Create a predictor with the audio preset
//! config = kizzasi.Config.audio(44100)
//! predictor = kizzasi.Predictor(config)
//!
//! # Single-step prediction
//! import numpy as np
//! inp = np.array([0.5], dtype=np.float32)
//! out = predictor.step(inp)
//!
//! # Multi-step prediction (autoregressive)
//! steps = predictor.predict_n(inp, n_steps=10)
//!
//! # Reset internal state
//! predictor.reset()
//!
//! # Multi-model ensemble
//! ensemble = kizzasi.EnsemblePredictor(config, n_models=3, voting="average")
//! out = ensemble.step(inp)
//!
//! # Cached / SIMD-accelerated predictor
//! opt = kizzasi.OptimizedPredictor(config, cache_ttl_ms=500)
//! out = opt.step(inp)
//! ```
//!
//! ## Module layout
//!
//! - [`config`]       — `PyModelType`, `PyKizzasiConfig`.
//! - [`predictor`]    — `PyPredictor`, `PyConstraintSpec`, guardrail helpers.
//! - [`ensemble`]     — `PyEnsemblePredictor` (multi-model voting).
//! - [`optimized`]    — `PyOptimizedPredictor` (workspace pool, SIMD, LRU cache).
//! - [`lora`]         — `PyLoRAAdapter` (low-rank adaptation for fine-tuning).
//! - [`sampling`]     — `PySamplingConfig`, `PySampler` (greedy / temperature /
//!   top-k / top-p strategies).
//! - [`beam_search`]  — `PyBeamSearch`, `PyConstrainedBeamSearch`,
//!   `PyRejectionSampler` (beam search and rejection sampling with Python
//!   callable constraints).

#![deny(warnings)]
#![deny(clippy::all)]

use pyo3::prelude::*;

mod beam_search;
mod config;
mod ensemble;
mod lora;
mod optimized;
mod predictor;
mod sampling;

use beam_search::{PyBeamSearch, PyConstrainedBeamSearch, PyRejectionSampler};
use config::{PyKizzasiConfig, PyModelType};
use ensemble::PyEnsemblePredictor;
use lora::PyLoRAAdapter;
use optimized::PyOptimizedPredictor;
use predictor::{PyConstraintSpec, PyPredictor};
use sampling::{PySampler, PySamplingConfig};

#[pymodule]
#[pyo3(name = "kizzasi")]
fn kizzasi_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKizzasiConfig>()?;
    m.add_class::<PyPredictor>()?;
    m.add_class::<PyConstraintSpec>()?;
    m.add_class::<PyModelType>()?;
    m.add_class::<PyEnsemblePredictor>()?;
    m.add_class::<PyOptimizedPredictor>()?;
    m.add_class::<PyLoRAAdapter>()?;
    m.add_class::<PySamplingConfig>()?;
    m.add_class::<PySampler>()?;
    m.add_class::<PyBeamSearch>()?;
    m.add_class::<PyConstrainedBeamSearch>()?;
    m.add_class::<PyRejectionSampler>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__doc__", "Kizzasi AGSP — PyO3 Python bindings")?;
    Ok(())
}
