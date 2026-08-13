//! Low-Rank Adaptation (LoRA) Python wrapper.
//!
//! Exposes [`PyLoRAAdapter`] — a thin facade over [`kizzasi_core::lora::LoRAAdapter`]
//! that lets Python users:
//! * construct an adapter with a chosen rank / alpha / dropout,
//! * add per-module `LoRALayer`s from a base weight matrix supplied as a NumPy
//!   array,
//! * run a `forward` pass on a single named module to apply the low-rank
//!   correction to an input vector,
//! * merge / unmerge LoRA contributions into the base weights in place,
//! * introspect parameter counts and per-module names.
//!
//! The wrapper deliberately keeps the surface area narrow: more advanced
//! features such as safetensors loading or custom target-module lists are
//! delegated to the core Rust API.
//!
//! ## Example
//!
//! ```python
//! import numpy as np
//! import kizzasi
//!
//! adapter = kizzasi.LoRAAdapter("my_adapter", rank=8, alpha=16.0)
//! base = np.random.randn(64, 128).astype(np.float32)
//! adapter.add_layer("layer_1", base)
//! out = adapter.forward("layer_1", np.random.randn(128).astype(np.float32))
//! ```

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scirs2_numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};

use kizzasi_core::lora::{LoRAAdapter, LoRAConfig, LoRALayer};
use scirs2_core::ndarray::{Array1, Array2};

use crate::predictor::to_py_err;

/// Low-Rank Adaptation adapter manager.
///
/// Wraps [`LoRAAdapter`] together with its [`LoRAConfig`]. Layers are
/// added one-at-a-time with their base weight matrix; the adapter then
/// owns an independently trainable low-rank correction (A, B) for each
/// registered module name.
///
/// Predictions go through [`forward`], which applies the LoRA correction
/// on top of the base weight (`y = W x + alpha/r * B (A x)`, or
/// `y = W x + alpha/r * B (A dropout(x))` while the adapter is in training
/// mode -- see [`train`](Self::train) -- and `dropout > 0`; adapters start
/// in evaluation mode, where `dropout` is always a no-op). Calling
/// [`merge_all`] folds every LoRA contribution into the base weights so
/// later `forward` calls become pure matrix multiplications;
/// [`unmerge_all`] reverses the operation. Merging is a static transform of
/// the trained weights, not a forward pass, so it is never subject to
/// dropout regardless of training mode -- and once merged, `forward` no
/// longer has a separate LoRA path to apply dropout to at all. Calling
/// `train()` and then `merge_all()` followed by `forward()` is therefore
/// deterministic; this is correct (not a sign dropout stopped working).
#[pyclass(name = "LoRAAdapter")]
pub struct PyLoRAAdapter {
    inner: LoRAAdapter,
}

#[pymethods]
impl PyLoRAAdapter {
    /// Build a new LoRA adapter.
    ///
    /// Parameters:
    /// - `name`: opaque identifier (used in `__repr__` and error messages).
    /// - `rank`: low-rank dimension; must be `> 0`.
    /// - `alpha`: scaling factor; must be `> 0` (effective scale = `alpha / rank`).
    /// - `dropout`: per-layer dropout probability in `[0, 1)`; default `0.0`.
    ///   Inert until [`train`](Self::train) is called -- the adapter (and
    ///   every layer it holds) starts in evaluation mode, where `forward`
    ///   ignores `dropout` entirely and stays fully deterministic no matter
    ///   what it is set to.
    #[new]
    #[pyo3(signature = (name, rank, alpha, dropout = 0.0))]
    pub fn new(name: String, rank: usize, alpha: f32, dropout: f32) -> PyResult<Self> {
        let config = LoRAConfig::new(rank, alpha).with_dropout(dropout);
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: LoRAAdapter::new(name, config),
        })
    }

    /// Register a new LoRA layer for a given module name.
    ///
    /// `base_weight` is the original (out_features, in_features) matrix the
    /// LoRA correction will be applied on top of. A fresh `(A, B)` pair is
    /// initialised: `A` with random values and `B` with zeros, so the initial
    /// effective weight equals `base_weight`.
    pub fn add_layer(
        &mut self,
        module_name: String,
        base_weight: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<()> {
        let arr: Array2<f32> = base_weight.as_array().to_owned();
        let layer = LoRALayer::new(self.inner.config.clone(), arr).map_err(to_py_err)?;
        self.inner.add_layer(module_name, layer);
        Ok(())
    }

    /// Apply the LoRA-augmented forward pass for a single registered module.
    ///
    /// Returns a float32 NumPy array of length `out_features`.
    pub fn forward<'py>(
        &self,
        py: Python<'py>,
        module: &str,
        input: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let layer =
            self.inner.layers.get(module).ok_or_else(|| {
                PyValueError::new_err(format!("LoRA module not found: '{}'", module))
            })?;
        let x: Array1<f32> = input.as_array().to_owned();
        let y = layer.forward(&x).map_err(to_py_err)?;
        Ok(y.into_pyarray(py))
    }

    /// Fold every LoRA correction into the base weights in-place.
    pub fn merge_all(&mut self) -> PyResult<()> {
        self.inner.merge_all().map_err(to_py_err)
    }

    /// Reverse [`merge_all`]: subtract every LoRA correction back out.
    pub fn unmerge_all(&mut self) -> PyResult<()> {
        self.inner.unmerge_all().map_err(to_py_err)
    }

    /// Switch every registered layer -- and any layer added afterwards --
    /// into training mode: `forward` will stochastically zero elements of
    /// the LoRA input path with probability `dropout` (rescaling survivors
    /// by `1 / (1 - dropout)`), so consecutive calls with the same input
    /// can now return different results wherever `dropout > 0`. Adapters
    /// start in evaluation mode (see [`eval`](Self::eval)).
    pub fn train(&mut self) {
        self.inner.train();
    }

    /// Switch every registered layer back into evaluation mode: `forward`
    /// becomes fully deterministic again and ignores `dropout` entirely.
    /// This is the default mode for a freshly constructed adapter.
    ///
    /// (This is the standard ML train/eval mode toggle -- mirroring e.g.
    /// PyTorch's `Module.eval()` -- not the `eval`/code-execution builtin
    /// found in dynamic languages; it flips a flag and runs no code.)
    pub fn eval(&mut self) {
        self.inner.eval();
    }

    /// Whether [`train`](Self::train) was called more recently than
    /// [`eval`](Self::eval). Adapters start in evaluation mode, so this is
    /// `False` until `train()` is called.
    #[getter]
    pub fn is_training(&self) -> bool {
        self.inner.is_training()
    }

    /// Total number of trainable LoRA parameters across all registered modules.
    pub fn total_parameters(&self) -> usize {
        self.inner.total_parameters()
    }

    /// Average per-module ratio of LoRA-trainable parameters vs the underlying
    /// base weights. `0.0` if no layers have been registered.
    pub fn avg_parameter_ratio(&self) -> f32 {
        self.inner.avg_parameter_ratio()
    }

    /// List of currently registered module names (insertion order is *not*
    /// preserved, since the underlying storage is a `HashMap`).
    pub fn module_names(&self) -> Vec<String> {
        self.inner.layers.keys().cloned().collect()
    }

    /// Adapter name supplied at construction.
    #[getter]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Low-rank dimension.
    #[getter]
    pub fn rank(&self) -> usize {
        self.inner.config.rank
    }

    /// Scaling factor (alpha; effective scale = `alpha / rank`).
    #[getter]
    pub fn alpha(&self) -> f32 {
        self.inner.config.alpha
    }

    /// Dropout probability used when initialising layers. Inert unless
    /// [`train`](Self::train) has been called -- see [`is_training`](Self::is_training).
    #[getter]
    pub fn dropout(&self) -> f32 {
        self.inner.config.dropout
    }

    /// Number of registered LoRA layers.
    #[getter]
    pub fn num_layers(&self) -> usize {
        self.inner.layers.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "LoRAAdapter(name='{}', rank={}, alpha={}, layers={})",
            self.inner.name,
            self.inner.config.rank,
            self.inner.config.alpha,
            self.inner.layers.len(),
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __len__(&self) -> usize {
        self.inner.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_construct() {
        let adapter = PyLoRAAdapter::new("a".to_string(), 4, 8.0, 0.1).expect("adapter");
        assert_eq!(adapter.name(), "a");
        assert_eq!(adapter.rank(), 4);
        assert!((adapter.alpha() - 8.0).abs() < 1e-6);
        assert!((adapter.dropout() - 0.1).abs() < 1e-6);
        assert_eq!(adapter.num_layers(), 0);
        assert_eq!(adapter.total_parameters(), 0);
        assert!(adapter.avg_parameter_ratio().abs() < 1e-6);
    }

    #[test]
    fn test_lora_add_and_forward() {
        let mut adapter = PyLoRAAdapter::new("b".to_string(), 2, 4.0, 0.0).expect("adapter");
        // Use the public Rust API directly because PyReadonlyArray<...> needs
        // the Python GIL to construct. The wrapper around `add_layer` /
        // `forward` is a near-1:1 forward to the inner Rust calls, so this
        // exercises the same code path as PyO3 would.
        let base = Array2::<f32>::from_elem((8, 16), 0.05);
        let layer = LoRALayer::new(adapter.inner.config.clone(), base).expect("layer");
        adapter.inner.add_layer("layer_1".to_string(), layer);

        assert_eq!(adapter.num_layers(), 1);
        let names = adapter.module_names();
        assert_eq!(names.len(), 1);
        assert_eq!(names[0], "layer_1");

        // total_parameters: rank * (in + out) = 2 * (16 + 8) = 48
        assert_eq!(adapter.total_parameters(), 48);

        // forward via the inner LoRA layer
        let x = Array1::<f32>::from_elem(16, 0.5);
        let y = adapter
            .inner
            .layers
            .get("layer_1")
            .expect("layer present")
            .forward(&x)
            .expect("forward");
        assert_eq!(y.len(), 8);
        for v in y.iter() {
            assert!(v.is_finite(), "lora output not finite: {}", v);
        }
    }

    #[test]
    fn test_lora_merge_unmerge() {
        let mut adapter = PyLoRAAdapter::new("c".to_string(), 2, 4.0, 0.0).expect("adapter");
        let base = Array2::<f32>::from_elem((4, 4), 0.25);
        let layer = LoRALayer::new(adapter.inner.config.clone(), base).expect("layer");
        adapter.inner.add_layer("m".to_string(), layer);

        // Capture the effective weight before any merge.
        let pre_merge = adapter
            .inner
            .layers
            .get("m")
            .expect("layer")
            .get_effective_weight();

        // merge -> unmerge should round-trip the effective weight.
        adapter.merge_all().expect("merge");
        adapter.unmerge_all().expect("unmerge");

        let post_unmerge = adapter
            .inner
            .layers
            .get("m")
            .expect("layer")
            .get_effective_weight();

        for (a, b) in pre_merge.iter().zip(post_unmerge.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "effective weight drifted after merge/unmerge: {} vs {}",
                a,
                b,
            );
        }
    }

    #[test]
    fn test_lora_invalid_rank() {
        // rank == 0 is rejected by LoRAConfig::validate
        let res = PyLoRAAdapter::new("bad".to_string(), 0, 8.0, 0.0);
        assert!(res.is_err(), "rank=0 should be rejected");
    }

    #[test]
    fn test_lora_invalid_alpha() {
        // alpha <= 0 is rejected by LoRAConfig::validate
        let res = PyLoRAAdapter::new("bad".to_string(), 4, 0.0, 0.0);
        assert!(res.is_err(), "alpha=0 should be rejected");
        let res = PyLoRAAdapter::new("bad".to_string(), 4, -1.0, 0.0);
        assert!(res.is_err(), "alpha<0 should be rejected");
    }

    #[test]
    fn test_lora_invalid_dropout() {
        // dropout outside [0, 1) is rejected
        let res = PyLoRAAdapter::new("bad".to_string(), 4, 8.0, 1.0);
        assert!(res.is_err(), "dropout=1.0 should be rejected");
        let res = PyLoRAAdapter::new("bad".to_string(), 4, 8.0, -0.1);
        assert!(res.is_err(), "dropout<0 should be rejected");
    }

    #[test]
    fn test_lora_train_eval_plumbing() {
        let mut adapter = PyLoRAAdapter::new("plumbing".to_string(), 2, 4.0, 0.3).expect("adapter");
        assert!(!adapter.is_training());

        adapter.train();
        assert!(adapter.is_training());

        adapter.eval();
        assert!(!adapter.is_training());
    }

    // Regression for the medium bug where `dropout` was accepted, validated,
    // and exposed as a property (`PyLoRAAdapter::dropout`) but never
    // consulted by `forward`: `dropout=0.3` and `dropout=0.0` produced
    // byte-identical output regardless of training/evaluation mode. As in
    // `kizzasi-core`'s equivalent tests, a freshly added layer's `lora_b`
    // starts at all zeros (see `LoRALayer::new`), which would make the LoRA
    // path -- and therefore any dropout applied to its input -- invisible
    // in `forward`'s output no matter how dropout is wired up. `set_lora_b`
    // is used first so these tests actually exercise the dropout-masked
    // path instead of passing vacuously.
    //
    // `lora_a` is deliberately left at its natural random initialisation
    // (never overridden) and the input is 64-wide at `dropout=0.5`:
    // `dropout=0.5` minimises the chance that two independent Bernoulli
    // masks coincide per element (`keep_prob^2 + (1-keep_prob)^2` is
    // minimised at `keep_prob=0.5`, unlike e.g. `dropout=0.9` where masks
    // mostly agree on "everything dropped"); at 64 independent elements the
    // chance two draws' masks coincide exactly is `0.5^64`, and a random
    // (not hand-picked) `lora_a` makes any *other* collision astronomically
    // unlikely too.
    fn lora_dropout_test_layer(dropout: f32) -> LoRALayer {
        let rank = 4;
        let out_features = 6;
        let in_features = 64;
        let config = LoRAConfig::new(rank, 8.0).with_dropout(dropout);
        let base = Array2::<f32>::from_shape_fn((out_features, in_features), |(i, j)| {
            (i as f32 + j as f32) * 0.001
        });
        let mut layer = LoRALayer::new(config, base).expect("layer");
        layer
            .set_lora_b(Array2::<f32>::from_shape_fn(
                (out_features, rank),
                |(i, j)| 0.1 + 0.03 * (i as f32) - 0.017 * (j as f32),
            ))
            .expect("set_lora_b");
        layer
    }

    #[test]
    fn test_lora_dropout_inactive_in_eval_mode() {
        let mut adapter =
            PyLoRAAdapter::new("eval_dropout".to_string(), 4, 8.0, 0.5).expect("adapter");
        let layer = lora_dropout_test_layer(0.5);
        adapter.inner.add_layer("m".to_string(), layer);
        assert!(!adapter.is_training());

        let x = Array1::<f32>::from_shape_fn(64, |i| (i as f32) * 0.01 + 0.1);
        let first = adapter
            .inner
            .layers
            .get("m")
            .expect("layer present")
            .forward(&x)
            .expect("forward");
        for _ in 0..20 {
            let repeat = adapter
                .inner
                .layers
                .get("m")
                .expect("layer present")
                .forward(&x)
                .expect("forward");
            assert_eq!(first, repeat, "eval-mode forward must be deterministic");
        }
    }

    #[test]
    fn test_lora_dropout_active_in_training_mode() {
        let mut adapter =
            PyLoRAAdapter::new("train_dropout".to_string(), 4, 8.0, 0.5).expect("adapter");
        let layer = lora_dropout_test_layer(0.5);
        adapter.inner.add_layer("m".to_string(), layer);

        adapter.train();
        assert!(adapter.is_training());
        assert!(adapter.inner.layers["m"].is_training());

        let x = Array1::<f32>::from_elem(64, 1.0);
        let first = adapter
            .inner
            .layers
            .get("m")
            .expect("layer present")
            .forward(&x)
            .expect("forward");
        let mut saw_difference = false;
        for _ in 0..20 {
            let repeat = adapter
                .inner
                .layers
                .get("m")
                .expect("layer present")
                .forward(&x)
                .expect("forward");
            if repeat != first {
                saw_difference = true;
                break;
            }
        }
        assert!(
            saw_difference,
            "training-mode dropout should make forward stochastic once B is nonzero"
        );
    }

    #[test]
    fn test_lora_add_layer_after_train_starts_training() {
        let mut adapter = PyLoRAAdapter::new("late_add".to_string(), 2, 4.0, 0.5).expect("adapter");
        adapter.train();

        let base = Array2::<f32>::from_elem((4, 4), 0.1);
        let layer = LoRALayer::new(adapter.inner.config.clone(), base).expect("layer");
        adapter.inner.add_layer("late".to_string(), layer);

        assert!(
            adapter.inner.layers["late"].is_training(),
            "a layer added after train() must also start in training mode"
        );
    }

    #[test]
    fn test_lora_avg_parameter_ratio() {
        let mut adapter = PyLoRAAdapter::new("d".to_string(), 4, 8.0, 0.0).expect("adapter");
        let base = Array2::<f32>::from_elem((64, 32), 0.1);
        let layer = LoRALayer::new(adapter.inner.config.clone(), base).expect("layer");
        adapter.inner.add_layer("x".to_string(), layer);

        // LoRA params = 4 * (64 + 32) = 384; base = 64 * 32 = 2048; ratio = 0.1875
        let ratio = adapter.avg_parameter_ratio();
        assert!((ratio - 0.1875).abs() < 1e-5);
    }

    #[test]
    fn test_lora_repr() {
        let adapter = PyLoRAAdapter::new("r".to_string(), 8, 16.0, 0.0).expect("adapter");
        let r = adapter.__repr__();
        assert!(r.contains("LoRAAdapter("));
        assert!(r.contains("name='r'"));
        assert!(r.contains("rank=8"));
        assert!(r.contains("layers=0"));
    }
}
