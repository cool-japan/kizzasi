//! Configuration and model-type Python wrappers.
//!
//! This module exposes:
//! - [`PyModelType`] — selector for the underlying SSM architecture.
//! - [`PyKizzasiConfig`] — configuration object with builder-style presets
//!   (`audio`, `robotics`, `sensor`, `lightweight`).
//!
//! These are pure data types — they hold no Rust handles and are cheap to
//! clone, which lets them be shared across multiple predictors (ensembles,
//! optimized predictors, …).

use pyo3::prelude::*;

use kizzasi_core::{KizzasiConfig, ModelType};

/// Selector for the underlying SSM architecture.
#[pyclass(name = "ModelType", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyModelType {
    inner: ModelType,
}

impl PyModelType {
    fn from_inner(inner: ModelType) -> Self {
        Self { inner }
    }

    /// Canonical lowercase name matching [`PyKizzasiConfig::parse_model_type`]'s
    /// accepted values (`"mamba"`, `"mamba2"`, `"s4"`, `"rwkv"`).
    pub(crate) fn canonical_name(&self) -> &'static str {
        match self.inner {
            ModelType::Mamba => "mamba",
            ModelType::Mamba2 => "mamba2",
            ModelType::S4 => "s4",
            ModelType::Rwkv => "rwkv",
        }
    }
}

/// Accepts either a `str` (parsed case-insensitively, matching
/// [`PyKizzasiConfig::parse_model_type`]) or a [`PyModelType`] instance
/// anywhere a model type is expected (`Config(...)`'s `model_type` keyword
/// argument and the `Config.model_type` property setter).
///
/// Before this type existed, `Config` only accepted `model_type` as a plain
/// `str`, so `kizzasi.ModelType` — registered, documented under its own
/// README heading, and returned by e.g. `ModelType.MAMBA2` — was a dead
/// class: passing it anywhere a model type was expected raised `TypeError`
/// at the call site, or silently stored an unusable value that only failed
/// later inside `parse_model_type`.
pub(crate) enum ModelTypeArg {
    Str(String),
    Enum(PyModelType),
}

impl ModelTypeArg {
    /// Canonical lowercase form, valid input to [`PyKizzasiConfig::parse_model_type`].
    /// A `Str` variant is passed through unchanged (still validated later by
    /// `parse_model_type`, preserving the existing lazy-validation behaviour
    /// of the `model_type: String` field); an `Enum` variant is always
    /// already-valid by construction.
    fn canonical(&self) -> String {
        match self {
            Self::Str(s) => s.clone(),
            Self::Enum(mt) => mt.canonical_name().to_string(),
        }
    }
}

impl<'py> FromPyObject<'_, 'py> for ModelTypeArg {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(mt) = obj.extract::<PyModelType>() {
            return Ok(Self::Enum(mt));
        }
        if let Ok(s) = obj.extract::<String>() {
            return Ok(Self::Str(s));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "model_type must be a str or a kizzasi.ModelType instance",
        ))
    }
}

#[pymethods]
impl PyModelType {
    #[classattr]
    #[allow(non_snake_case)]
    fn MAMBA() -> Self {
        Self::from_inner(ModelType::Mamba)
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn MAMBA2() -> Self {
        Self::from_inner(ModelType::Mamba2)
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn S4() -> Self {
        Self::from_inner(ModelType::S4)
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn RWKV() -> Self {
        Self::from_inner(ModelType::Rwkv)
    }

    fn __repr__(&self) -> String {
        match self.inner {
            ModelType::Mamba => "ModelType.MAMBA".to_string(),
            ModelType::Mamba2 => "ModelType.MAMBA2".to_string(),
            ModelType::S4 => "ModelType.S4".to_string(),
            ModelType::Rwkv => "ModelType.RWKV".to_string(),
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Configuration for a Kizzasi AGSP predictor.
#[pyclass(name = "Config", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyKizzasiConfig {
    #[pyo3(get, set)]
    pub input_dim: usize,
    #[pyo3(get, set)]
    pub output_dim: usize,
    #[pyo3(get, set)]
    pub hidden_dim: usize,
    #[pyo3(get, set)]
    pub num_layers: usize,
    #[pyo3(get, set)]
    pub state_dim: usize,
    /// Context-window size, in steps. Accepted, range-checked (`> 0` by
    /// [`Self::validate_dimensions`]), forwarded to
    /// [`KizzasiConfig::context_window`], and readable back via this
    /// property -- but it does not bound memory or computation today: the
    /// underlying `SelectiveSSM` engine keeps a fixed-size per-layer hidden
    /// state that is completely independent of this value (see
    /// `kizzasi_core::ssm::SelectiveSSM`'s `context_window` doc comment).
    /// Two `Predictor`s built from configs that differ only in
    /// `context_window` use identical memory and produce identical
    /// predictions for the same input sequence.
    #[pyo3(get, set)]
    pub context_window: usize,
    /// SSM architecture name, always stored in canonical lowercase form
    /// ("mamba", "mamba2", "s4", "rwkv") regardless of whether it was set
    /// from a `str` or a [`PyModelType`] instance.
    ///
    /// No `#[pyo3(get, set)]` here: the Python-visible `model_type`
    /// property has hand-written accessors below so its setter can accept
    /// [`ModelTypeArg`] (`str | ModelType`) instead of only `str`.
    pub model_type: String,
}

/// Upper bound on the total number of `f32` parameters a [`PyKizzasiConfig`]
/// may request when building a predictor (summed across the embedding,
/// hidden state, and every layer's A/B/C/D/output-projection matrices — see
/// [`PyKizzasiConfig::checked_total_parameters`]). At 4 bytes/element this
/// caps worst-case allocation at ~4 GiB: far more than any legitimate
/// real-time signal-prediction model needs, but low enough that a request
/// like `hidden_dim=10**9` is rejected with a catchable `PyValueError`
/// instead of aborting the process when the allocator gives up.
const MAX_TOTAL_PARAMETERS: usize = 1 << 30;

fn dimension_cap_error() -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "requested model size exceeds the supported parameter cap of {} \
         elements (~{} GiB of f32); reduce input_dim, output_dim, \
         hidden_dim, state_dim, and/or num_layers",
        MAX_TOTAL_PARAMETERS,
        (MAX_TOTAL_PARAMETERS as f64 * 4.0) / (1024.0 * 1024.0 * 1024.0),
    ))
}

impl PyKizzasiConfig {
    /// Parse the textual model-type identifier. Case-insensitive.
    pub(crate) fn parse_model_type(s: &str) -> Result<ModelType, PyErr> {
        match s.to_lowercase().as_str() {
            "mamba" => Ok(ModelType::Mamba),
            "mamba2" => Ok(ModelType::Mamba2),
            "s4" => Ok(ModelType::S4),
            "rwkv" => Ok(ModelType::Rwkv),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown model_type '{}'. Valid options: mamba, mamba2, s4, rwkv",
                other
            ))),
        }
    }

    /// Total `f32` element count the engine allocates for this
    /// configuration, mirroring `SelectiveSSM::new`'s actual allocations
    /// (`kizzasi-core/src/ssm.rs`) plus `ContinuousEmbedding::new`'s
    /// (`kizzasi-core/src/embedding.rs`) and the layered `HiddenState`'s
    /// (`kizzasi-core/src/state.rs`):
    ///
    /// - embedding weights: `input_dim * hidden_dim`
    /// - embedding bias: `hidden_dim`
    /// - hidden state: `num_layers * hidden_dim * state_dim`
    /// - per-layer A, B, C matrices: `num_layers * 3 * hidden_dim * state_dim`
    /// - per-layer D vectors: `num_layers * hidden_dim`
    /// - per-layer Δ (time-step) low-rank projection, `dt_proj_down`
    ///   `(hidden_dim, dt_rank)` and `dt_proj_up` `(dt_rank, hidden_dim)`:
    ///   `num_layers * 2 * hidden_dim * dt_rank`
    /// - output projection: `hidden_dim * output_dim`
    ///
    /// `dt_rank` has no Python-visible knob — `to_core_config` never calls
    /// `KizzasiConfig::dt_rank(...)`, so the engine always uses
    /// `kizzasi_core::KizzasiConfig::default()`'s value. Reading it from
    /// there (rather than hardcoding the constant here) keeps this in sync
    /// if that default ever changes, without needing to touch the
    /// `kizzasi-core` crate.
    ///
    /// The Δ-projection term was previously missing from this accounting
    /// entirely. Because `dt_rank` defaults to 8 (a *fixed* per-layer
    /// coefficient on `hidden_dim`, independent of `state_dim`), a config
    /// with a large `hidden_dim` but minimal `state_dim`/`num_layers` could
    /// pass the old, incomplete total comfortably under
    /// [`MAX_TOTAL_PARAMETERS`] while the real allocation — once
    /// `dt_proj_down`/`dt_proj_up` are included — exceeded it several times
    /// over, reproducing the exact unbounded-allocation abort this cap
    /// exists to prevent. See
    /// `test_to_core_config_rejects_huge_hidden_dim_via_dt_projection`.
    ///
    /// Uses `checked_mul`/`checked_add` throughout, so an absurd input
    /// (e.g. `usize::MAX`) reports the cap error via overflow (`None`)
    /// rather than wrapping into a small, wrongly-accepted total.
    fn checked_total_parameters(&self) -> Option<usize> {
        let dt_rank = KizzasiConfig::default().get_dt_rank().max(1);

        let embedding = self.input_dim.checked_mul(self.hidden_dim)?;
        let embedding_bias = self.hidden_dim;
        let per_layer_state = self.hidden_dim.checked_mul(self.state_dim)?;
        let hidden_state = per_layer_state.checked_mul(self.num_layers)?;
        let abc_matrices = per_layer_state
            .checked_mul(3)?
            .checked_mul(self.num_layers)?;
        let d_vectors = self.hidden_dim.checked_mul(self.num_layers)?;
        let dt_projection = self
            .hidden_dim
            .checked_mul(dt_rank)?
            .checked_mul(2)?
            .checked_mul(self.num_layers)?;
        let output_proj = self.hidden_dim.checked_mul(self.output_dim)?;

        embedding
            .checked_add(embedding_bias)?
            .checked_add(hidden_state)?
            .checked_add(abc_matrices)?
            .checked_add(d_vectors)?
            .checked_add(dt_projection)?
            .checked_add(output_proj)
    }

    /// Reject dimensions that would make `SelectiveSSM::new` abort the
    /// process instead of returning a catchable error: any dimension of
    /// zero (silently producing a predictor whose weights — and therefore
    /// every prediction — are empty), or a total allocation request above
    /// [`MAX_TOTAL_PARAMETERS`] (which, unchecked, can make the global
    /// allocator abort the whole interpreter on a single oversized
    /// `hidden_dim` from pure user input).
    fn validate_dimensions(&self) -> Result<(), PyErr> {
        for (name, value) in [
            ("input_dim", self.input_dim),
            ("output_dim", self.output_dim),
            ("hidden_dim", self.hidden_dim),
            ("num_layers", self.num_layers),
            ("state_dim", self.state_dim),
            ("context_window", self.context_window),
        ] {
            if value == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{name} must be > 0"
                )));
            }
        }

        match self.checked_total_parameters() {
            Some(total) if total <= MAX_TOTAL_PARAMETERS => Ok(()),
            _ => Err(dimension_cap_error()),
        }
    }

    /// Convert this Python config into the internal [`KizzasiConfig`].
    pub(crate) fn to_core_config(&self) -> Result<KizzasiConfig, PyErr> {
        let model_type = Self::parse_model_type(&self.model_type)?;
        self.validate_dimensions()?;
        Ok(KizzasiConfig::new()
            .model_type(model_type)
            .input_dim(self.input_dim)
            .output_dim(self.output_dim)
            .hidden_dim(self.hidden_dim)
            .num_layers(self.num_layers)
            .state_dim(self.state_dim)
            .context_window(self.context_window))
    }
}

impl PyKizzasiConfig {
    /// Rust-side constructor (`model_type` as a plain canonical `String`).
    /// The Python-visible `__new__` is [`Self::py_new`], which additionally
    /// accepts a [`PyModelType`] and normalizes it before delegating here.
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        state_dim: usize,
        context_window: usize,
        model_type: String,
    ) -> Self {
        Self {
            input_dim,
            output_dim,
            hidden_dim,
            num_layers,
            state_dim,
            context_window,
            model_type,
        }
    }
}

#[pymethods]
impl PyKizzasiConfig {
    #[new]
    #[pyo3(signature = (
        input_dim,
        output_dim,
        hidden_dim = 256,
        num_layers = 4,
        state_dim = 16,
        context_window = 8192,
        model_type = ModelTypeArg::Str("mamba2".to_string())
    ))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        input_dim: usize,
        output_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        state_dim: usize,
        context_window: usize,
        model_type: ModelTypeArg,
    ) -> Self {
        Self::new(
            input_dim,
            output_dim,
            hidden_dim,
            num_layers,
            state_dim,
            context_window,
            model_type.canonical(),
        )
    }

    /// SSM architecture name: `"mamba"`, `"mamba2"`, `"s4"`, or `"rwkv"`.
    /// Always reads back as the canonical lowercase `str`, even if last set
    /// from a [`PyModelType`] instance.
    #[getter(model_type)]
    fn get_model_type(&self) -> String {
        self.model_type.clone()
    }

    /// Accepts either a `str` or a `kizzasi.ModelType` instance (e.g.
    /// `cfg.model_type = kizzasi.ModelType.RWKV`); both are normalized to
    /// the canonical lowercase `str` form on assignment.
    #[setter(model_type)]
    fn set_model_type(&mut self, value: ModelTypeArg) {
        self.model_type = value.canonical();
    }

    /// Audio signal prediction preset.
    ///
    /// `context_window` scales with `sample_rate` so every rate covers the
    /// same real-time horizon as the 44.1 kHz reference default (8192
    /// samples ≈ 185.8 ms), rounded up to the next power of two. Previously
    /// `sample_rate` was accepted but silently discarded — every call
    /// returned byte-identical output regardless of its value.
    #[staticmethod]
    #[pyo3(signature = (sample_rate = 44100))]
    pub fn audio(sample_rate: u32) -> PyResult<Self> {
        if sample_rate == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sample_rate must be > 0",
            ));
        }
        const REFERENCE_RATE: u64 = 44_100;
        const REFERENCE_WINDOW: u64 = 8192;
        let scaled = (u64::from(sample_rate) * REFERENCE_WINDOW).div_ceil(REFERENCE_RATE);
        let context_window = scaled.max(1).next_power_of_two() as usize;
        Ok(Self {
            input_dim: 1,
            output_dim: 1,
            hidden_dim: 256,
            num_layers: 4,
            state_dim: 16,
            context_window,
            model_type: "mamba2".to_string(),
        })
    }

    /// Robotics / control-loop preset.
    #[staticmethod]
    pub fn robotics(state_dim: usize, action_dim: usize) -> Self {
        Self {
            input_dim: state_dim,
            output_dim: action_dim,
            hidden_dim: 128,
            num_layers: 3,
            state_dim: 8,
            context_window: 1024,
            model_type: "mamba2".to_string(),
        }
    }

    /// Multi-sensor fusion preset.
    #[staticmethod]
    pub fn sensor(num_sensors: usize) -> Self {
        Self {
            input_dim: num_sensors,
            output_dim: num_sensors,
            hidden_dim: 64,
            num_layers: 2,
            state_dim: 8,
            context_window: 2048,
            model_type: "mamba2".to_string(),
        }
    }

    /// Lightweight embedded / edge preset.
    #[staticmethod]
    pub fn lightweight(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            hidden_dim: 32,
            num_layers: 1,
            state_dim: 4,
            context_window: 512,
            model_type: "mamba".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Config(input_dim={}, output_dim={}, hidden_dim={}, num_layers={}, \
             state_dim={}, context_window={}, model_type='{}')",
            self.input_dim,
            self.output_dim,
            self.hidden_dim,
            self.num_layers,
            self.state_dim,
            self.context_window,
            self.model_type
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
    fn test_config_creation_defaults() {
        let cfg = PyKizzasiConfig::new(4, 4, 256, 4, 16, 8192, "mamba2".to_string());
        assert_eq!(cfg.input_dim, 4);
        assert_eq!(cfg.output_dim, 4);
        assert_eq!(cfg.hidden_dim, 256);
        assert_eq!(cfg.num_layers, 4);
        assert_eq!(cfg.state_dim, 16);
        assert_eq!(cfg.context_window, 8192);
        assert_eq!(cfg.model_type, "mamba2");
    }

    #[test]
    fn test_audio_preset_dims() {
        let cfg = PyKizzasiConfig::audio(44100).expect("audio preset");
        assert_eq!(cfg.input_dim, 1);
        assert_eq!(cfg.output_dim, 1);
        assert_eq!(cfg.hidden_dim, 256);
        assert_eq!(cfg.num_layers, 4);
        // Reference rate: context_window must stay byte-identical to the
        // pre-fix constant default, so this preset's documented behaviour
        // at the default sample_rate does not change.
        assert_eq!(cfg.context_window, 8192);
    }

    // Regression for the medium bug where `sample_rate` was accepted and
    // then discarded (`let _ = sample_rate;`): every call returned a
    // byte-identical `Config` regardless of the argument.
    #[test]
    fn test_audio_preset_sample_rate_scales_context_window() {
        let reference = PyKizzasiConfig::audio(44100).expect("reference preset");
        let lower = PyKizzasiConfig::audio(16000).expect("16kHz preset");
        let higher = PyKizzasiConfig::audio(96000).expect("96kHz preset");
        assert_ne!(
            lower.context_window, reference.context_window,
            "sample_rate must affect context_window"
        );
        assert!(lower.context_window < reference.context_window);
        assert!(higher.context_window > reference.context_window);
        // Every returned context_window is a power of two (the SSM's
        // preferred shape) and covers at least the requested horizon.
        assert!(lower.context_window.is_power_of_two());
        assert!(higher.context_window.is_power_of_two());
    }

    #[test]
    fn test_audio_preset_rejects_zero_sample_rate() {
        assert!(PyKizzasiConfig::audio(0).is_err());
    }

    #[test]
    fn test_robotics_preset_dims() {
        let cfg = PyKizzasiConfig::robotics(6, 4);
        assert_eq!(cfg.input_dim, 6);
        assert_eq!(cfg.output_dim, 4);
        assert_eq!(cfg.hidden_dim, 128);
        assert_eq!(cfg.num_layers, 3);
    }

    #[test]
    fn test_sensor_preset_dims() {
        let cfg = PyKizzasiConfig::sensor(10);
        assert_eq!(cfg.input_dim, 10);
        assert_eq!(cfg.output_dim, 10);
        assert_eq!(cfg.hidden_dim, 64);
        assert_eq!(cfg.num_layers, 2);
    }

    #[test]
    fn test_lightweight_preset_dims() {
        let cfg = PyKizzasiConfig::lightweight(2, 3);
        assert_eq!(cfg.input_dim, 2);
        assert_eq!(cfg.output_dim, 3);
        assert_eq!(cfg.hidden_dim, 32);
        assert_eq!(cfg.num_layers, 1);
        assert_eq!(cfg.model_type, "mamba");
    }

    #[test]
    fn test_config_repr() {
        let cfg = PyKizzasiConfig::audio(44100).expect("audio preset");
        let r = cfg.__repr__();
        assert!(r.contains("Config("));
        assert!(r.contains("input_dim=1"));
        assert!(r.contains("mamba2"));
    }

    #[test]
    fn test_core_config_conversion_valid() {
        let cfg = PyKizzasiConfig::new(3, 3, 64, 2, 8, 512, "mamba2".to_string());
        let result = cfg.to_core_config();
        assert!(result.is_ok(), "Expected Ok, got {:?}", result.err());
        let core = result.expect("core config");
        assert_eq!(core.get_input_dim(), 3);
        assert_eq!(core.get_output_dim(), 3);
    }

    #[test]
    fn test_core_config_conversion_invalid_model_type() {
        let cfg = PyKizzasiConfig::new(1, 1, 32, 1, 4, 512, "transformer".to_string());
        assert!(cfg.to_core_config().is_err());
    }

    #[test]
    fn test_model_type_repr_round_trip() {
        // Each classattr returns a PyModelType; check repr matches.
        assert_eq!(PyModelType::MAMBA().__repr__(), "ModelType.MAMBA");
        assert_eq!(PyModelType::MAMBA2().__repr__(), "ModelType.MAMBA2");
        assert_eq!(PyModelType::S4().__repr__(), "ModelType.S4");
        assert_eq!(PyModelType::RWKV().__repr__(), "ModelType.RWKV");
    }

    #[test]
    fn test_parse_model_type_case_insensitive() {
        assert_eq!(
            PyKizzasiConfig::parse_model_type("MAMBA").expect("parse"),
            ModelType::Mamba
        );
        assert_eq!(
            PyKizzasiConfig::parse_model_type("Mamba2").expect("parse"),
            ModelType::Mamba2
        );
        assert!(PyKizzasiConfig::parse_model_type("nonexistent").is_err());
    }

    // Regression for the medium bug where `ModelType` was registered and
    // documented but no API accepted it: `py_new`'s `model_type` parameter
    // (and the `model_type` property setter) now accept a `PyModelType`
    // via `ModelTypeArg`, not just a `str`.
    #[test]
    fn test_model_type_arg_canonical_from_enum() {
        let arg = ModelTypeArg::Enum(PyModelType::RWKV());
        assert_eq!(arg.canonical(), "rwkv");
        let arg = ModelTypeArg::Enum(PyModelType::S4());
        assert_eq!(arg.canonical(), "s4");
    }

    #[test]
    fn test_model_type_arg_canonical_from_str() {
        let arg = ModelTypeArg::Str("Mamba2".to_string());
        // Str variant passes through unchanged (case as given); validation
        // happens later in `parse_model_type`, matching the pre-existing
        // lazy-validation behaviour of the plain-`String` path.
        assert_eq!(arg.canonical(), "Mamba2");
    }

    #[test]
    fn test_py_new_accepts_model_type_enum() {
        let cfg = PyKizzasiConfig::py_new(
            4,
            4,
            256,
            4,
            16,
            8192,
            ModelTypeArg::Enum(PyModelType::RWKV()),
        );
        assert_eq!(cfg.model_type, "rwkv");
        assert!(cfg.to_core_config().is_ok());
    }

    #[test]
    fn test_set_model_type_accepts_enum() {
        let mut cfg = PyKizzasiConfig::new(2, 2, 32, 1, 4, 256, "mamba2".to_string());
        cfg.set_model_type(ModelTypeArg::Enum(PyModelType::S4()));
        assert_eq!(cfg.get_model_type(), "s4");
        assert_eq!(cfg.model_type, "s4");
    }

    // Regression for the critical robustness bug where an unvalidated
    // `hidden_dim` (e.g. `10**9`) reached `SelectiveSSM::new`'s
    // `Array2::from_shape_fn` allocations directly, risking a process abort
    // on allocation failure instead of a catchable Python exception.
    #[test]
    fn test_to_core_config_rejects_huge_hidden_dim() {
        let cfg = PyKizzasiConfig::new(1, 1, 1_000_000_000, 4, 16, 8192, "mamba2".to_string());
        assert!(cfg.to_core_config().is_err());
    }

    // Same failure mode via a huge `input_dim` alone (dominates through the
    // embedding matrix, not the hidden_dim*state_dim terms) — regression
    // for undercounting the checked total.
    #[test]
    fn test_to_core_config_rejects_huge_input_dim() {
        let cfg = PyKizzasiConfig::new(1_000_000_000, 1, 256, 4, 16, 8192, "mamba2".to_string());
        assert!(cfg.to_core_config().is_err());
    }

    // Regression for a second undercounting bug in the same critical fix:
    // `checked_total_parameters` originally omitted `SelectiveSSM::new`'s
    // per-layer `dt_proj_down`/`dt_proj_up` (Δ low-rank projection)
    // matrices entirely. `dt_rank` defaults to 8 in `kizzasi_core` and has
    // no Python-visible knob, so with `state_dim`/`num_layers`/`input_dim`/
    // `output_dim` all pinned at their allowed minimum of 1 and only
    // `hidden_dim` large, the *old* accounting totalled `7 * hidden_dim`
    // (comfortably under the 2^30 cap at hidden_dim=100_000_000: 700M), but
    // the *real* allocation — once the Δ-projection matrices are included —
    // is `23 * hidden_dim` (2.3B elements, ~9.2 GB of f32), more than 2x
    // over the cap. This exact shape would have silently reproduced the
    // process-abort-on-allocation vulnerability
    // `test_to_core_config_rejects_huge_hidden_dim` above exists to close.
    #[test]
    fn test_to_core_config_rejects_huge_hidden_dim_via_dt_projection() {
        let cfg = PyKizzasiConfig::new(1, 1, 100_000_000, 1, 1, 64, "mamba2".to_string());
        assert!(
            cfg.to_core_config().is_err(),
            "hidden_dim=1e8 must be rejected once the Δ-projection matrices \
             are counted, even though state_dim/num_layers are minimal"
        );
    }

    // The Δ-projection accounting must not become *over*-conservative: this
    // config's real allocation (including dt_proj_down/up) is well under
    // the cap and must still be accepted.
    #[test]
    fn test_to_core_config_accepts_model_with_dt_projection_counted() {
        let cfg = PyKizzasiConfig::new(64, 64, 4096, 4, 128, 8192, "mamba2".to_string());
        assert!(cfg.to_core_config().is_ok());
    }

    #[test]
    fn test_to_core_config_rejects_zero_dimensions() {
        let zero_input = PyKizzasiConfig::new(0, 1, 32, 1, 4, 64, "mamba2".to_string());
        assert!(zero_input.to_core_config().is_err());
        let zero_hidden = PyKizzasiConfig::new(1, 1, 0, 1, 4, 64, "mamba2".to_string());
        assert!(zero_hidden.to_core_config().is_err());
        let zero_layers = PyKizzasiConfig::new(1, 1, 32, 0, 4, 64, "mamba2".to_string());
        assert!(zero_layers.to_core_config().is_err());
        let zero_state = PyKizzasiConfig::new(1, 1, 32, 1, 0, 64, "mamba2".to_string());
        assert!(zero_state.to_core_config().is_err());
    }

    #[test]
    fn test_to_core_config_accepts_reasonable_large_model() {
        // hidden_dim=4096, state_dim=128, num_layers=24 is already a very
        // large real-time SSM by this crate's standards; must stay under
        // the cap so legitimate large configs are not rejected.
        let cfg = PyKizzasiConfig::new(256, 256, 4096, 24, 128, 8192, "mamba2".to_string());
        assert!(cfg.to_core_config().is_ok());
    }
}
