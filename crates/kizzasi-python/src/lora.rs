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
use scirs2_numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};

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
/// on top of the base weight (`y = W x + alpha/r * B (A x)`). Calling
/// [`merge_all`] folds every LoRA contribution into the base weights so
/// later `forward` calls become pure matrix multiplications;
/// [`unmerge_all`] reverses the operation.
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
        Ok(y.to_pyarray(py))
    }

    /// Fold every LoRA correction into the base weights in-place.
    pub fn merge_all(&mut self) -> PyResult<()> {
        self.inner.merge_all().map_err(to_py_err)
    }

    /// Reverse [`merge_all`]: subtract every LoRA correction back out.
    pub fn unmerge_all(&mut self) -> PyResult<()> {
        self.inner.unmerge_all().map_err(to_py_err)
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

    /// Dropout probability used when initialising layers.
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
