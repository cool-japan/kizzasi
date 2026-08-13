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
//! - [`mulaw`]        — `PyMuLawCodec` (μ-law audio companding, mirroring
//!   `kizzasi-tokenizer`'s WASM binding surface).

#![deny(warnings)]
#![deny(clippy::all)]

use pyo3::prelude::*;

mod beam_search;
mod config;
mod ensemble;
mod lora;
mod mulaw;
mod optimized;
mod predictor;
mod sampling;

use beam_search::{PyBeamSearch, PyConstrainedBeamSearch, PyRejectionSampler};
use config::{PyKizzasiConfig, PyModelType};
use ensemble::PyEnsemblePredictor;
use lora::PyLoRAAdapter;
use mulaw::PyMuLawCodec;
use optimized::PyOptimizedPredictor;
use predictor::{PyConstraintSpec, PyPredictor};
use sampling::{PySampler, PySamplingConfig};

// The compiled extension is installed as `kizzasi._kizzasi` (see
// `module-name = "kizzasi._kizzasi"` in pyproject.toml) and `python/kizzasi/__init__.py`
// imports everything from `._kizzasi`. CPython's `ExtensionFileLoader` derives the
// `PyInit_<name>` entry-point symbol it looks up from the *last* component of that
// dotted path, i.e. `PyInit__kizzasi` (two leading underscores: one from the `PyInit_`
// prefix, one from the module's own leading underscore) — NOT `PyInit_kizzasi`. The
// `#[pyo3(name = "_kizzasi")]` override below must keep matching `module-name`'s last
// segment; renaming one without the other silently breaks `import kizzasi` (CPython
// raises `ImportError: dynamic module does not define module export function`).
#[pymodule]
#[pyo3(name = "_kizzasi")]
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
    m.add_class::<PyMuLawCodec>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__doc__", "Kizzasi AGSP — PyO3 Python bindings")?;
    Ok(())
}

#[cfg(test)]
mod module_export_tests {
    use super::*;

    /// Regression for the critical bug where `python/kizzasi/__init__.py`
    /// re-exported only 2 of the 12 classes `kizzasi_module` registered
    /// (13 now, after `MuLawCodec`) — `kizzasi.EnsemblePredictor`,
    /// `.OptimizedPredictor`, etc. raised `AttributeError` from Python
    /// despite being documented and registered here.
    ///
    /// This builds the *actual* module via `kizzasi_module` (not a
    /// hand-maintained list of names that could itself drift out of sync)
    /// and checks every non-dunder name it registers appears in
    /// `__init__.py`'s source, read via `include_str!` so this runs under
    /// plain `cargo nextest` with no built wheel required. A test that only
    /// checked "every name *in* `__all__` resolves" (the subset direction)
    /// would not have caught the original bug — the broken `__all__ =
    /// ["Config", "Predictor", "__version__"]` was itself a subset that
    /// resolved just fine; the bug was what `__all__` was *missing*. Only
    /// this "every registered name appears in `__init__.py`" direction
    /// detects that.
    #[test]
    fn test_init_py_reexports_every_registered_class() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_kizzasi").expect("construct test module");
            kizzasi_module(&module).expect("kizzasi_module registration must succeed");
            let dir = module.dir().expect("dir() on the constructed module");
            let init_py = include_str!("../python/kizzasi/__init__.py");

            let mut public_names_checked = 0usize;
            for item in dir.iter() {
                let name: String = item.extract().expect("dir() entries are str");
                // Dunders (`__version__`, `__doc__`, `__name__`, `__loader__`, ...)
                // are not classes and are not expected to be re-exported by name
                // in the same way; `__init__.py` only needs the real pyclasses.
                if name.starts_with('_') {
                    continue;
                }
                public_names_checked += 1;
                assert!(
                    init_py.contains(name.as_str()),
                    "'{name}' is registered by kizzasi_module (lib.rs) but does not \
                     appear anywhere in python/kizzasi/__init__.py's source — it \
                     will raise AttributeError as `kizzasi.{name}` from Python. \
                     This is exactly the shape of the critical __init__.py \
                     export-gap bug.",
                );
            }
            assert!(
                public_names_checked >= 13,
                "expected at least the 13 known pyclasses to be registered and \
                 checked, only found {public_names_checked} — the dir() walk \
                 above may not be discovering registrations correctly",
            );
        });
    }
}
