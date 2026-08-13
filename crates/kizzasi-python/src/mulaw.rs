//! μ-law audio codec Python wrapper.
//!
//! Exposes [`PyMuLawCodec`] — a thin facade over
//! [`kizzasi_tokenizer::MuLawCodec`], the same codec already exposed to
//! JavaScript via `kizzasi-tokenizer`'s `wasm` feature
//! (`WasmMuLawCodec` in `kizzasi-tokenizer/src/wasm_bindings.rs`).
//!
//! μ-law (ITU-T G.711) is a logarithmic companding scheme that preserves
//! dynamic range better than linear quantization for quiet sounds — the
//! standard codec for telephony and WaveNet-style audio models.
//!
//! `kizzasi-tokenizer` is already part of this crate's build graph
//! transitively (both `kizzasi` and `kizzasi-inference` depend on it), so
//! this module adds no new third-party dependency — see `Cargo.toml`.

use pyo3::prelude::*;
use scirs2_numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};

use kizzasi_tokenizer::{MuLawCodec, SignalTokenizer};
use scirs2_core::ndarray::Array1;

use crate::predictor::to_py_err;

/// μ-law companding codec for audio signal quantization.
///
/// μ-law encoding is a logarithmic quantization scheme that preserves
/// dynamic range better than linear quantization, especially for quiet
/// sounds (ITU-T G.711).
///
/// Example
/// -------
/// .. code-block:: python
///
///     import numpy as np
///     import kizzasi
///
///     codec = kizzasi.MuLawCodec(bits=8)
///     signal = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
///
///     # Discrete quantization levels in [0, vocab_size), as int32 ...
///     levels = codec.quantize_array(signal)
///     back = codec.dequantize_array(levels)
///     # ... or the same levels as float32 via the SignalTokenizer trait
///     # convention shared with every other tokenizer in kizzasi-tokenizer
///     # (encode/decode operate on token ids, not the raw companded value).
///     tokens = codec.encode(signal)
///     reconstructed = codec.decode(tokens)
#[pyclass(name = "MuLawCodec")]
pub struct PyMuLawCodec {
    inner: MuLawCodec,
}

#[pymethods]
impl PyMuLawCodec {
    /// Construct a codec with the standard `mu = (2**bits) - 1`.
    ///
    /// Parameters
    /// ----------
    /// bits:
    ///     Quantization bit depth. Must be in `1..=16`. Default `8`
    ///     (mu = 255, the standard telephony/G.711 depth).
    #[new]
    #[pyo3(signature = (bits = 8))]
    pub fn new(bits: u8) -> PyResult<Self> {
        Ok(Self {
            inner: MuLawCodec::try_new(bits).map_err(to_py_err)?,
        })
    }

    /// Construct a codec with an explicit `mu` parameter instead of the
    /// standard `(2**bits) - 1`.
    ///
    /// Parameters
    /// ----------
    /// mu:
    ///     Companding parameter. Must be finite and > 0.
    /// bits:
    ///     Quantization bit depth used for `quantize`/`dequantize` and
    ///     `vocab_size`. Must be in `1..=16`. Default `8`.
    #[staticmethod]
    #[pyo3(signature = (mu, bits = 8))]
    pub fn with_mu(mu: f32, bits: u8) -> PyResult<Self> {
        Ok(Self {
            inner: MuLawCodec::try_with_mu(mu, bits).map_err(to_py_err)?,
        })
    }

    /// Quantize a single sample in `[-1.0, 1.0]` to an integer level in
    /// `[0, vocab_size)`. Out-of-range input is clamped, matching the
    /// underlying codec.
    pub fn quantize(&self, x: f32) -> i32 {
        self.inner.quantize(x)
    }

    /// Dequantize an integer level back to a float sample in `[-1.0, 1.0]`.
    pub fn dequantize(&self, level: i32) -> f32 {
        self.inner.dequantize(level)
    }

    /// Vectorized [`Self::quantize`] over a full signal.
    ///
    /// Parameters
    /// ----------
    /// signal:
    ///     Float32 array of samples in `[-1.0, 1.0]`.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of dtype int32, same shape as `signal`, each element
    /// in `[0, vocab_size)`.
    pub fn quantize_array<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<'_, f32>,
    ) -> Bound<'py, PyArray1<i32>> {
        let arr = signal.as_array();
        let levels: Vec<i32> = arr.iter().map(|&x| self.inner.quantize(x)).collect();
        levels.into_pyarray(py)
    }

    /// Vectorized [`Self::dequantize`] over a full array of integer levels.
    ///
    /// Parameters
    /// ----------
    /// levels:
    ///     int32 array of quantization levels.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of dtype float32, same shape as `levels`, each element
    /// in `[-1.0, 1.0]`.
    pub fn dequantize_array<'py>(
        &self,
        py: Python<'py>,
        levels: PyReadonlyArray1<'_, i32>,
    ) -> Bound<'py, PyArray1<f32>> {
        let arr = levels.as_array();
        let samples: Vec<f32> = arr.iter().map(|&lvl| self.inner.dequantize(lvl)).collect();
        samples.into_pyarray(py)
    }

    /// μ-law encode a full signal via [`SignalTokenizer::encode`].
    ///
    /// Returns discrete quantization levels in `[0, vocab_size)`, as
    /// `float32` token ids — the same convention every tokenizer in
    /// `kizzasi-tokenizer` shares (matches [`Self::quantize_array`]'s
    /// values exactly, just as `float32` instead of `int32`; use
    /// `quantize_array` directly if you want the more precisely-typed
    /// `int32` levels without going through the trait).
    pub fn encode<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let arr: Array1<f32> = signal.as_array().to_owned();
        let out = self.inner.encode(&arr).map_err(to_py_err)?;
        Ok(out.into_pyarray(py))
    }

    /// Inverse of [`Self::encode`] via [`SignalTokenizer::decode`]: takes
    /// `float32` token ids (as produced by `encode`, rounded to the nearest
    /// integer level internally) and returns the reconstructed signal.
    pub fn decode<'py>(
        &self,
        py: Python<'py>,
        tokens: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let arr: Array1<f32> = tokens.as_array().to_owned();
        let out = self.inner.decode(&arr).map_err(to_py_err)?;
        Ok(out.into_pyarray(py))
    }

    /// Bit depth this codec was constructed with.
    #[getter]
    pub fn bits(&self) -> u8 {
        self.inner.bits()
    }

    /// The `mu` companding parameter.
    #[getter]
    pub fn mu(&self) -> f32 {
        self.inner.mu()
    }

    /// Number of discrete quantization levels (`2 ** bits`); the exclusive
    /// upper bound of [`Self::quantize`]'s / [`Self::quantize_array`]'s output.
    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.inner.levels()
    }

    fn __repr__(&self) -> String {
        format!(
            "MuLawCodec(bits={}, mu={})",
            self.inner.bits(),
            self.inner.mu()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mulaw_new_default_bits() {
        let codec = PyMuLawCodec::new(8).expect("codec");
        assert_eq!(codec.bits(), 8);
        assert_eq!(codec.vocab_size(), 256);
        assert!((codec.mu() - 255.0).abs() < 1e-6);
    }

    #[test]
    fn test_mulaw_new_rejects_zero_bits() {
        assert!(PyMuLawCodec::new(0).is_err());
    }

    #[test]
    fn test_mulaw_new_rejects_too_many_bits() {
        assert!(PyMuLawCodec::new(17).is_err());
    }

    #[test]
    fn test_mulaw_with_mu_custom() {
        let codec = PyMuLawCodec::with_mu(100.0, 8).expect("codec");
        assert!((codec.mu() - 100.0).abs() < 1e-6);
        assert_eq!(codec.bits(), 8);
    }

    #[test]
    fn test_mulaw_with_mu_rejects_nonfinite() {
        assert!(PyMuLawCodec::with_mu(f32::NAN, 8).is_err());
        assert!(PyMuLawCodec::with_mu(f32::INFINITY, 8).is_err());
        assert!(PyMuLawCodec::with_mu(0.0, 8).is_err());
        assert!(PyMuLawCodec::with_mu(-1.0, 8).is_err());
    }

    #[test]
    fn test_mulaw_quantize_dequantize_roundtrip() {
        let codec = PyMuLawCodec::new(8).expect("codec");
        for x in [-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
            let level = codec.quantize(x);
            assert!((0..256).contains(&level));
            let back = codec.dequantize(level);
            assert!(
                (back - x).abs() < 0.05,
                "roundtrip failed for {}: got {}",
                x,
                back
            );
        }
    }

    #[test]
    fn test_mulaw_quantize_midpoint() {
        let codec = PyMuLawCodec::new(8).expect("codec");
        assert_eq!(codec.quantize(0.0), 128);
        assert_eq!(codec.quantize(-1.0), 0);
        assert_eq!(codec.quantize(1.0), 255);
    }

    #[test]
    fn test_mulaw_repr() {
        let codec = PyMuLawCodec::new(8).expect("codec");
        let r = codec.__repr__();
        assert!(r.contains("MuLawCodec("));
        assert!(r.contains("bits=8"));
    }

    /// `MuLawCodec` must stay `Send` (and `Sync`, since this pyclass is not
    /// `unsendable`) for the `#[pyclass]` macro to accept it.
    #[test]
    fn test_mulaw_codec_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MuLawCodec>();
    }

    #[test]
    fn test_mulaw_quantize_array_crosses_pyo3_boundary() {
        use scirs2_numpy::{PyArray1, PyArrayMethods};
        Python::initialize();
        Python::attach(|py| {
            let codec = PyMuLawCodec::new(8).expect("codec");
            let py_arr = PyArray1::from_vec(py, vec![-1.0_f32, 0.0, 1.0]);
            let levels = codec.quantize_array(py, py_arr.readonly());
            let levels_ro = levels.readonly();
            let view = levels_ro.as_array();
            assert_eq!(view.to_vec(), vec![0, 128, 255]);

            let levels_arr = PyArray1::from_vec(py, vec![0_i32, 128, 255]);
            let back = codec.dequantize_array(py, levels_arr.readonly());
            let back_ro = back.readonly();
            let back_view = back_ro.as_array();
            assert!((back_view[0] - (-1.0)).abs() < 0.05);
            assert!((back_view[1] - 0.0).abs() < 0.05);
            assert!((back_view[2] - 1.0).abs() < 0.05);
        });
    }

    // `encode`/`decode` follow the `SignalTokenizer` trait convention
    // (discrete levels as float32 token ids), matching `quantize_array`'s
    // values exactly — see the doc comments on `Self::encode`/`Self::decode`.
    #[test]
    fn test_mulaw_encode_decode_crosses_pyo3_boundary() {
        use scirs2_numpy::{PyArray1, PyArrayMethods};
        Python::initialize();
        Python::attach(|py| {
            let codec = PyMuLawCodec::new(8).expect("codec");
            let py_arr = PyArray1::from_vec(py, vec![0.0_f32, 0.5, -0.5, 1.0, -1.0]);
            let tokens = codec
                .encode(py, py_arr.readonly())
                .expect("encode via pymethod");
            let tokens_ro = tokens.readonly();
            for v in tokens_ro.as_array().iter() {
                assert!(
                    (0.0..256.0).contains(v),
                    "token value out of vocab range: {}",
                    v
                );
            }
            let reconstructed = codec.decode(py, tokens_ro).expect("decode via pymethod");
            let reconstructed_ro = reconstructed.readonly();
            let original = [0.0_f32, 0.5, -0.5, 1.0, -1.0];
            for (orig, dec) in original.iter().zip(reconstructed_ro.as_array().iter()) {
                assert!(
                    (orig - dec).abs() < 0.05,
                    "roundtrip failed: {} vs {}",
                    orig,
                    dec
                );
            }
        });
    }
}
