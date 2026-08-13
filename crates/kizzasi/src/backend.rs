//! Model backend dispatch for [`crate::Kizzasi`].
//!
//! [`KizzasiConfig::get_model_type`] selects which architecture actually runs.
//! Every variant of [`ModelType`] maps to a distinct implementation:
//!
//! | `ModelType` | Engine | Crate |
//! |---|---|---|
//! | [`ModelType::Mamba2`] | [`SelectiveSSM`] — selective scan with per-channel Δ | `kizzasi-core` |
//! | [`ModelType::Mamba`] | `kizzasi_model::mamba::Mamba` — gated block + causal conv | `kizzasi-model` |
//! | [`ModelType::S4`] | `kizzasi_model::s4::S4D` — diagonal structured SSM | `kizzasi-model` |
//! | [`ModelType::Rwkv`] | `kizzasi_model::rwkv::Rwkv` — multi-head linear attention | `kizzasi-model` |
//!
//! ## Why `Mamba2` is served by `SelectiveSSM`
//!
//! [`SelectiveSSM`] is the crate's native streaming engine and the only
//! backend that is `Clone` + `Serialize`, which is what
//! [`crate::Kizzasi::fork`] and [`crate::FullStateCheckpoint`] require. It
//! implements a selective scan with an input-dependent per-channel Δ. It is
//! *not* the chunked SSD / multi-head formulation of the Mamba-2 paper — for
//! that, use `kizzasi_model::mamba2::Mamba2` directly. The other three
//! variants dispatch to `kizzasi-model`, whose models are neither `Clone` nor
//! able to export weights in memory, so `fork()` and full-state checkpoints
//! return a typed error for them rather than silently producing an unrelated
//! randomly-initialised model.
//!
//! ## State-capture completeness
//!
//! [`Backend::snapshot_state`] / [`Backend::restore_state`] round-trip the
//! full recurrent state for `SelectiveSSM`, `Mamba` (SSM state plus causal
//! convolution history) and `Rwkv` (WKV numerator/denominator/max plus both
//! token-shift buffers). `S4D` is the one exception: `kizzasi-model` exposes
//! only its SSM state, not the 3-tap causal-convolution history, so a
//! snapshot/restore around a profiling run leaves that short history advanced.
//!
//! ## Architectures not reachable through `ModelType`
//!
//! `kizzasi-model` also ships `Transformer`, `S5`, `Rwkv7`, `H3`, `MoE` and
//! others. [`ModelType`] has no variant for them and adding one would be a
//! breaking change for every exhaustive `match` on it outside this crate
//! (`kizzasi-python` has two). Construct those models directly from
//! `kizzasi-model`.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use kizzasi_core::{HiddenState, KizzasiConfig, ModelType, SelectiveSSM, SignalPredictor};
use kizzasi_model::mamba::{Mamba, MambaConfig};
use kizzasi_model::rwkv::{Rwkv, RwkvConfig};
use kizzasi_model::s4::{S4Config, S4D};
use kizzasi_model::AutoregressiveModel;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rng, RngExt};
use serde::{Deserialize, Serialize};

use crate::error::{KizzasiError, KizzasiResult};

/// On-disk format version written by [`Backend::save_weights`].
const WEIGHT_FORMAT_VERSION: u32 = 1;

/// Convert a `kizzasi-model` error into the facade error type.
fn model_err(context: &str, err: kizzasi_model::ModelError) -> KizzasiError {
    KizzasiError::inference(format!("{context}: {err}"))
}

/// The recurrent state of a backend, captured so it can be restored later.
///
/// Used by profiling paths that must not leave synthetic samples in a live
/// predictor's hidden state.
#[derive(Debug, Clone)]
pub enum StateSnapshot {
    /// State of the [`SelectiveSSM`] engine (the step counter lives inside
    /// [`HiddenState`], so it is restored with it).
    Selective(Box<HiddenState>),
    /// Per-layer states of a `kizzasi-model` architecture.
    ///
    /// The step counter is tracked by the facade rather than by the model, so
    /// it has to travel with the snapshot: restoring the layer states alone
    /// would leave `Kizzasi::step_count` reporting the profiling run's count
    /// instead of the caller's.
    Layered {
        states: Vec<HiddenState>,
        step_count: usize,
    },
}

/// Weight payload, tagged by the engine that produced it.
#[derive(Debug, Serialize, Deserialize)]
enum WeightPayload {
    /// Full serialisation of the selective-scan engine.
    SelectiveSsm(Box<SelectiveSSM>),
    /// Flat `name -> row-major values` map, the `kizzasi-model` convention.
    Tensors(BTreeMap<String, Vec<f32>>),
}

/// Self-describing weight file written by [`crate::Kizzasi::save_weights`] and
/// read back when `weights_path` is set on a configuration.
#[derive(Debug, Serialize, Deserialize)]
struct WeightFile {
    format_version: u32,
    model_type: String,
    input_dim: usize,
    output_dim: usize,
    hidden_dim: usize,
    state_dim: usize,
    num_layers: usize,
    /// Resolved architecture geometry.
    ///
    /// `num_heads` / `head_dim` are *derived* when the configuration leaves
    /// them unset, and `num_heads * head_dim == hidden_dim` holds for every
    /// admissible split — so the per-tensor element counts are identical
    /// across splits and a shape check alone cannot tell them apart. Recording
    /// the resolved values is what stops a 8x4 checkpoint from loading
    /// silently into a 4x8 layout.
    #[serde(default = "one")]
    num_heads: usize,
    #[serde(default = "one")]
    head_dim: usize,
    #[serde(default = "one")]
    expansion_factor: usize,
    /// Facade-owned readout projection, row-major `(native_out, output_dim)`.
    #[serde(default)]
    readout: Option<Vec<f32>>,
    payload: WeightPayload,
}

/// Serde default for the geometry fields.
fn one() -> usize {
    1
}

/// The architecture geometry actually used to build a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResolvedGeometry {
    num_heads: usize,
    head_dim: usize,
    expansion_factor: usize,
}

impl Default for ResolvedGeometry {
    fn default() -> Self {
        Self {
            num_heads: 1,
            head_dim: 1,
            expansion_factor: 1,
        }
    }
}

/// One of the `kizzasi-model` architectures reachable through [`ModelType`].
///
/// Held as a concrete enum rather than `Box<dyn AutoregressiveModel>` because
/// the in-memory weight loader (`load_weights_map`, which reports how many
/// tensors it applied) is an inherent method on each model, not a trait item.
enum Architecture {
    Mamba(Box<Mamba>),
    S4d(Box<S4D>),
    Rwkv(Box<Rwkv>),
}

impl Architecture {
    fn as_predictor(&mut self) -> &mut dyn SignalPredictor {
        match self {
            Self::Mamba(model) => model.as_mut(),
            Self::S4d(model) => model.as_mut(),
            Self::Rwkv(model) => model.as_mut(),
        }
    }

    fn get_states(&self) -> Vec<HiddenState> {
        match self {
            Self::Mamba(model) => model.get_states(),
            Self::S4d(model) => model.get_states(),
            Self::Rwkv(model) => model.get_states(),
        }
    }

    fn set_states(&mut self, states: Vec<HiddenState>) -> Result<(), kizzasi_model::ModelError> {
        match self {
            Self::Mamba(model) => model.set_states(states),
            Self::S4d(model) => model.set_states(states),
            Self::Rwkv(model) => model.set_states(states),
        }
    }

    fn save_weights_json(&self, path: &Path) -> Result<(), kizzasi_model::ModelError> {
        match self {
            Self::Mamba(model) => model.save_weights_json(path),
            Self::S4d(model) => model.save_weights_json(path),
            Self::Rwkv(model) => model.save_weights_json(path),
        }
    }

    /// Apply a flat weight map, returning how many tensors were actually used.
    fn load_weights_map(
        &mut self,
        map: &HashMap<String, Vec<f32>>,
    ) -> Result<usize, kizzasi_model::ModelError> {
        match self {
            Self::Mamba(model) => model.load_weights_map(map),
            Self::S4d(model) => model.load_weights_map(map),
            Self::Rwkv(model) => model.load_weights_map(map),
        }
    }
}

/// A `kizzasi-model` architecture plus the facade-owned readout projection.
///
/// Opaque: it exists so [`Backend::Architecture`] can name a type. Everything
/// callers need is reachable through [`Backend`]'s methods.
pub struct ArchitectureBackend {
    model: Architecture,
    /// Geometry the model was actually built with, recorded in weight files.
    geometry: ResolvedGeometry,
    /// `(native_out_dim, output_dim)` projection, `None` when the model's
    /// native output width already equals `output_dim`.
    readout: Option<Array2<f32>>,
    /// Number of `step` calls since construction or the last `reset`.
    step_count: usize,
    context_window: usize,
}

/// The engine behind a [`crate::Kizzasi`] predictor.
pub enum Backend {
    /// `kizzasi-core`'s selective-scan engine (`ModelType::Mamba2`).
    Selective(Box<SelectiveSSM>),
    /// A `kizzasi-model` architecture (`ModelType::Mamba` / `S4` / `Rwkv`).
    Architecture(Box<ArchitectureBackend>),
}

impl std::fmt::Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Selective(_) => f.write_str("Backend::Selective"),
            Self::Architecture(_) => f.write_str("Backend::Architecture"),
        }
    }
}

/// Largest head count that divides `hidden_dim`, capped at `preferred`.
///
/// The multi-head architectures require `hidden_dim % num_heads == 0`; the
/// facade's `hidden_dim` is user-chosen and may not be divisible by the
/// architecture default, so pick the largest admissible value instead of
/// failing on a perfectly reasonable configuration.
fn default_num_heads(hidden_dim: usize, preferred: usize) -> usize {
    let mut heads = preferred.min(hidden_dim).max(1);
    while heads > 1 && !hidden_dim.is_multiple_of(heads) {
        heads -= 1;
    }
    heads
}

/// Reject an option that the requested engine cannot honour.
fn reject_unused(field: &str, engine: &str, alternatives: &str) -> KizzasiError {
    KizzasiError::config(format!(
        "{field} is not supported by the {engine} engine (selected by this ModelType); \
         remove it or select one of: {alternatives}"
    ))
}

fn validated_num_heads(config: &KizzasiConfig, preferred: usize) -> KizzasiResult<usize> {
    let hidden_dim = config.get_hidden_dim();
    match config.get_num_heads() {
        Some(0) => Err(KizzasiError::config("num_heads must be > 0")),
        Some(heads) => {
            if !hidden_dim.is_multiple_of(heads) {
                return Err(KizzasiError::config(format!(
                    "hidden_dim ({hidden_dim}) must be divisible by num_heads ({heads})"
                )));
            }
            Ok(heads)
        }
        None => Ok(default_num_heads(hidden_dim, preferred)),
    }
}

fn validated_head_dim(config: &KizzasiConfig, num_heads: usize) -> KizzasiResult<usize> {
    let hidden_dim = config.get_hidden_dim();
    let derived = (hidden_dim / num_heads.max(1)).max(1);
    match config.get_head_dim() {
        Some(0) => Err(KizzasiError::config("head_dim must be > 0")),
        Some(dim) if dim != derived => Err(KizzasiError::config(format!(
            "head_dim ({dim}) must equal hidden_dim / num_heads \
             ({hidden_dim} / {num_heads} = {derived})"
        ))),
        Some(dim) => Ok(dim),
        None => Ok(derived),
    }
}

impl Backend {
    /// Build the engine selected by `config.get_model_type()`.
    ///
    /// Weights are **not** loaded here; [`crate::Kizzasi::new`] applies
    /// `config.get_weights_path()` afterwards so a load failure is reported
    /// with the file path in context.
    pub fn from_config(config: &KizzasiConfig) -> KizzasiResult<Self> {
        match config.get_model_type() {
            ModelType::Mamba2 => {
                // The selective-scan engine has no gated expansion branch and
                // no heads: refuse those options rather than dropping them.
                if config.get_expansion_factor().is_some() {
                    return Err(reject_unused(
                        "expansion_factor",
                        "selective-scan (SelectiveSSM)",
                        "ModelType::Mamba",
                    ));
                }
                if config.get_num_heads().is_some() {
                    return Err(reject_unused(
                        "num_heads",
                        "selective-scan (SelectiveSSM)",
                        "ModelType::Rwkv",
                    ));
                }
                if config.get_head_dim().is_some() {
                    return Err(reject_unused(
                        "head_dim",
                        "selective-scan (SelectiveSSM)",
                        "ModelType::Rwkv",
                    ));
                }
                Ok(Self::Selective(Box::new(SelectiveSSM::new(
                    config.clone(),
                )?)))
            }
            ModelType::Mamba => {
                if config.get_num_heads().is_some() {
                    return Err(reject_unused(
                        "num_heads",
                        "Mamba (single-head gated block)",
                        "ModelType::Rwkv",
                    ));
                }
                if config.get_head_dim().is_some() {
                    return Err(reject_unused(
                        "head_dim",
                        "Mamba (single-head gated block)",
                        "ModelType::Rwkv",
                    ));
                }
                let expand_factor = match config.get_expansion_factor() {
                    Some(0) => return Err(KizzasiError::config("expansion_factor must be > 0")),
                    Some(factor) => factor,
                    None => 2,
                };
                let model_config = MambaConfig {
                    input_dim: config.get_input_dim(),
                    hidden_dim: config.get_hidden_dim(),
                    state_dim: config.get_state_dim(),
                    expand_factor,
                    conv_kernel_size: 4,
                    num_layers: config.get_num_layers(),
                    dropout: 0.0,
                    use_mamba2: false,
                };
                let model = Mamba::new(model_config)
                    .map_err(|e| model_err("failed to build the Mamba backend", e))?;
                Ok(Self::architecture(
                    config,
                    Architecture::Mamba(Box::new(model)),
                    ResolvedGeometry {
                        num_heads: 1,
                        head_dim: 1,
                        expansion_factor: expand_factor,
                    },
                ))
            }
            ModelType::S4 => {
                if config.get_expansion_factor().is_some() {
                    return Err(reject_unused(
                        "expansion_factor",
                        "S4D (diagonal structured SSM)",
                        "ModelType::Mamba",
                    ));
                }
                if config.get_num_heads().is_some() {
                    return Err(reject_unused(
                        "num_heads",
                        "S4D (diagonal structured SSM)",
                        "ModelType::Rwkv",
                    ));
                }
                if config.get_head_dim().is_some() {
                    return Err(reject_unused(
                        "head_dim",
                        "S4D (diagonal structured SSM)",
                        "ModelType::Rwkv",
                    ));
                }
                let model_config = S4Config {
                    input_dim: config.get_input_dim(),
                    hidden_dim: config.get_hidden_dim(),
                    state_dim: config.get_state_dim(),
                    num_layers: config.get_num_layers(),
                    dropout: 0.0,
                    dt_min: 0.001,
                    dt_max: 0.1,
                    use_diagonal: true,
                    use_rms_norm: true,
                };
                let model = S4D::new(model_config)
                    .map_err(|e| model_err("failed to build the S4D backend", e))?;
                Ok(Self::architecture(
                    config,
                    Architecture::S4d(Box::new(model)),
                    ResolvedGeometry::default(),
                ))
            }
            ModelType::Rwkv => {
                if config.get_expansion_factor().is_some() {
                    return Err(reject_unused(
                        "expansion_factor",
                        "RWKV (linear attention)",
                        "ModelType::Mamba",
                    ));
                }
                let num_heads = validated_num_heads(config, 8)?;
                let head_dim = validated_head_dim(config, num_heads)?;
                let model_config = RwkvConfig {
                    input_dim: config.get_input_dim(),
                    hidden_dim: config.get_hidden_dim(),
                    intermediate_dim: config.get_hidden_dim() * 4,
                    num_layers: config.get_num_layers(),
                    num_heads,
                    head_dim,
                    dropout: 0.0,
                    time_decay_init: -5.0,
                    use_rms_norm: true,
                };
                let model = Rwkv::new(model_config)
                    .map_err(|e| model_err("failed to build the RWKV backend", e))?;
                Ok(Self::architecture(
                    config,
                    Architecture::Rwkv(Box::new(model)),
                    ResolvedGeometry {
                        num_heads,
                        head_dim,
                        expansion_factor: 1,
                    },
                ))
            }
        }
    }

    /// Wrap a `kizzasi-model` architecture, adding a readout projection when
    /// the configured `output_dim` differs from the model's native width.
    fn architecture(
        config: &KizzasiConfig,
        model: Architecture,
        geometry: ResolvedGeometry,
    ) -> Self {
        let native_out = config.get_input_dim();
        let output_dim = config.get_output_dim();
        let readout = if native_out == output_dim {
            None
        } else {
            let mut generator = rng();
            let scale = (2.0 / (native_out + output_dim) as f32).sqrt();
            Some(Array2::from_shape_fn((native_out, output_dim), |_| {
                (generator.random::<f32>() - 0.5) * 2.0 * scale
            }))
        };

        Self::Architecture(Box::new(ArchitectureBackend {
            model,
            geometry,
            readout,
            step_count: 0,
            context_window: config.get_context_window(),
        }))
    }

    /// Build a backend around an already-constructed selective SSM.
    pub fn from_selective(ssm: SelectiveSSM) -> Self {
        Self::Selective(Box::new(ssm))
    }

    /// Human-readable engine name, used in error messages and docs.
    pub fn engine_name(&self) -> &'static str {
        match self {
            Self::Selective(_) => "SelectiveSSM",
            Self::Architecture(_) => "kizzasi-model architecture",
        }
    }

    /// Advance the model by one sample.
    pub fn step(&mut self, input: &Array1<f32>) -> KizzasiResult<Array1<f32>> {
        match self {
            Self::Selective(ssm) => Ok(ssm.step(input)?),
            Self::Architecture(arch) => {
                let raw = arch
                    .model
                    .as_predictor()
                    .step(input)
                    .map_err(|e| KizzasiError::inference(format!("model step failed: {e}")))?;
                arch.step_count += 1;
                match &arch.readout {
                    Some(projection) => {
                        if raw.len() != projection.shape()[0] {
                            return Err(KizzasiError::dimension_mismatch(
                                projection.shape()[0],
                                raw.len(),
                                "model output width does not match the readout projection",
                            ));
                        }
                        Ok(raw.dot(projection))
                    }
                    None => Ok(raw),
                }
            }
        }
    }

    /// Reset the recurrent state (weights are untouched).
    pub fn reset(&mut self) {
        match self {
            Self::Selective(ssm) => ssm.reset(),
            Self::Architecture(arch) => {
                arch.model.as_predictor().reset();
                arch.step_count = 0;
            }
        }
    }

    /// Configured context window.
    pub fn context_window(&self) -> usize {
        match self {
            Self::Selective(ssm) => ssm.context_window(),
            Self::Architecture(arch) => arch.context_window,
        }
    }

    /// Number of steps taken since construction or the last reset.
    pub fn step_count(&self) -> usize {
        match self {
            Self::Selective(ssm) => ssm.step_count(),
            Self::Architecture(arch) => arch.step_count,
        }
    }

    /// Borrow the selective SSM, if this backend is one.
    pub fn selective(&self) -> Option<&SelectiveSSM> {
        match self {
            Self::Selective(ssm) => Some(ssm),
            Self::Architecture(_) => None,
        }
    }

    /// Mutably borrow the selective SSM, if this backend is one.
    pub fn selective_mut(&mut self) -> Option<&mut SelectiveSSM> {
        match self {
            Self::Selective(ssm) => Some(ssm),
            Self::Architecture(_) => None,
        }
    }

    /// Copy the backend, weights included.
    ///
    /// Only the [`SelectiveSSM`] engine can be copied: the `kizzasi-model`
    /// architectures are neither `Clone` nor able to export their weights in
    /// memory, so "copying" them would mean re-randomising every parameter.
    pub fn try_clone(&self) -> KizzasiResult<Self> {
        match self {
            Self::Selective(ssm) => Ok(Self::Selective(ssm.clone())),
            Self::Architecture(_) => Err(KizzasiError::invalid_state_with_recovery(
                "this predictor's backend cannot be copied: the kizzasi-model architectures \
                 selected by ModelType::Mamba / S4 / Rwkv expose no in-memory weight export, \
                 so a copy would silently be a differently-initialised model",
                "use ModelType::Mamba2 (the SelectiveSSM engine) for fork() and full-state \
                 checkpoints, or persist with save_weights() and rebuild with weights_path",
            )),
        }
    }

    /// Capture the recurrent state so it can be restored after profiling.
    pub fn snapshot_state(&self) -> StateSnapshot {
        match self {
            Self::Selective(ssm) => StateSnapshot::Selective(Box::new(ssm.get_state().clone())),
            Self::Architecture(arch) => StateSnapshot::Layered {
                states: arch.model.get_states(),
                step_count: arch.step_count,
            },
        }
    }

    /// Restore a state previously captured by [`Self::snapshot_state`].
    pub fn restore_state(&mut self, snapshot: StateSnapshot) -> KizzasiResult<()> {
        match (self, snapshot) {
            (Self::Selective(ssm), StateSnapshot::Selective(state)) => {
                ssm.set_state(*state);
                Ok(())
            }
            (Self::Architecture(arch), StateSnapshot::Layered { states, step_count }) => {
                arch.model.set_states(states).map_err(|e| {
                    KizzasiError::invalid_state(format!("failed to restore model state: {e}"))
                })?;
                arch.step_count = step_count;
                Ok(())
            }
            _ => Err(KizzasiError::invalid_state(
                "state snapshot was captured from a different backend kind",
            )),
        }
    }

    /// Write every trainable parameter to `path` as a self-describing JSON
    /// weight file.
    ///
    /// The result is exactly what [`crate::KizzasiBuilder::weights_path`]
    /// expects; see [`Self::load_weights`].
    pub fn save_weights(&self, path: &Path, config: &KizzasiConfig) -> KizzasiResult<()> {
        let (payload, readout, geometry) = match self {
            Self::Selective(ssm) => (
                WeightPayload::SelectiveSsm(ssm.clone()),
                None,
                ResolvedGeometry::default(),
            ),
            Self::Architecture(arch) => {
                // The architectures only expose a file-based export, so write
                // their native map to `path` first and read it straight back to
                // build the envelope. No temporary file is involved: the final
                // content of `path` is the envelope written below.
                arch.model.save_weights_json(path).map_err(|e| {
                    model_err(
                        &format!("failed to export weights to {}", path.display()),
                        e,
                    )
                })?;
                let raw = std::fs::read_to_string(path).map_err(|e| {
                    KizzasiError::inference(format!(
                        "failed to re-read exported weights from {}: {e}",
                        path.display()
                    ))
                })?;
                let tensors: BTreeMap<String, Vec<f32>> =
                    serde_json::from_str(&raw).map_err(|e| {
                        KizzasiError::inference(format!(
                            "exported weights are not a flat weight map: {e}"
                        ))
                    })?;
                let readout = arch
                    .readout
                    .as_ref()
                    .map(|projection| projection.iter().copied().collect::<Vec<f32>>());
                (WeightPayload::Tensors(tensors), readout, arch.geometry)
            }
        };

        let file = WeightFile {
            format_version: WEIGHT_FORMAT_VERSION,
            model_type: format!("{:?}", config.get_model_type()),
            input_dim: config.get_input_dim(),
            output_dim: config.get_output_dim(),
            hidden_dim: config.get_hidden_dim(),
            state_dim: config.get_state_dim(),
            num_layers: config.get_num_layers(),
            num_heads: geometry.num_heads,
            head_dim: geometry.head_dim,
            expansion_factor: geometry.expansion_factor,
            readout,
            payload,
        };

        let encoded = serde_json::to_string(&file)
            .map_err(|e| KizzasiError::inference(format!("failed to serialize weights: {e}")))?;
        std::fs::write(path, encoded).map_err(|e| {
            KizzasiError::inference(format!("failed to write {}: {e}", path.display()))
        })
    }

    /// Load weights from `path` into this backend, replacing the random
    /// initialisation.
    ///
    /// Accepts either the JSON weight file written by [`Self::save_weights`]
    /// or, for the `kizzasi-model` architectures, a `.safetensors` checkpoint.
    /// Any failure — missing file, wrong engine, shape mismatch, or a tensor
    /// map that matches nothing in the model — is reported as an error; the
    /// backend is never left silently random-initialised.
    pub fn load_weights(&mut self, path: &Path, config: &KizzasiConfig) -> KizzasiResult<()> {
        if !path.exists() {
            return Err(KizzasiError::model_not_ready(
                format!("weights file not found: {}", path.display()),
                "check weights_path in the configuration, or omit it to use a \
                 randomly-initialised model",
            ));
        }

        let is_safetensors = path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"));

        if is_safetensors {
            return self.load_safetensors(path);
        }

        let raw = std::fs::read_to_string(path).map_err(|e| {
            KizzasiError::model_not_ready(
                format!("failed to read weights file {}: {e}", path.display()),
                "ensure the path points at a readable JSON weight file written by \
                 Kizzasi::save_weights",
            )
        })?;
        let file: WeightFile = serde_json::from_str(&raw).map_err(|e| {
            KizzasiError::model_not_ready(
                format!("{} is not a Kizzasi weight file: {e}", path.display()),
                "write it with Kizzasi::save_weights, or pass a .safetensors checkpoint \
                 for the Mamba / S4 / Rwkv backends",
            )
        })?;

        if file.format_version != WEIGHT_FORMAT_VERSION {
            return Err(KizzasiError::model_not_ready(
                format!(
                    "weight file {} has format version {}, this build reads version {}",
                    path.display(),
                    file.format_version,
                    WEIGHT_FORMAT_VERSION
                ),
                "regenerate the weight file with this version of Kizzasi",
            ));
        }

        let expected_type = format!("{:?}", config.get_model_type());
        if file.model_type != expected_type {
            return Err(KizzasiError::model_not_ready(
                format!(
                    "weight file {} was written for ModelType::{} but the configuration \
                     selects ModelType::{}",
                    path.display(),
                    file.model_type,
                    expected_type
                ),
                "set the same model_type that was used when the weights were saved",
            ));
        }

        check_dim("input_dim", file.input_dim, config.get_input_dim())?;
        check_dim("output_dim", file.output_dim, config.get_output_dim())?;
        check_dim("hidden_dim", file.hidden_dim, config.get_hidden_dim())?;
        check_dim("state_dim", file.state_dim, config.get_state_dim())?;
        check_dim("num_layers", file.num_layers, config.get_num_layers())?;

        // Head geometry is not implied by the dimensions above: every
        // admissible (num_heads, head_dim) split of the same hidden_dim
        // yields identical tensor element counts, so only an explicit
        // comparison prevents a silently mis-shaped load.
        let current_geometry = match self {
            Self::Architecture(arch) => arch.geometry,
            Self::Selective(_) => ResolvedGeometry::default(),
        };
        check_dim("num_heads", file.num_heads, current_geometry.num_heads)?;
        check_dim("head_dim", file.head_dim, current_geometry.head_dim)?;
        check_dim(
            "expansion_factor",
            file.expansion_factor,
            current_geometry.expansion_factor,
        )?;

        let engine_name = self.engine_name();
        match (self, file.payload) {
            (Self::Selective(target), WeightPayload::SelectiveSsm(loaded)) => {
                *target = loaded;
                Ok(())
            }
            (Self::Architecture(arch), WeightPayload::Tensors(tensors)) => {
                let map: HashMap<String, Vec<f32>> = tensors.into_iter().collect();
                apply_tensor_map(arch, &map, path)?;
                restore_readout(arch, file.readout.as_deref(), config)
            }
            _ => Err(KizzasiError::model_not_ready(
                format!(
                    "weight file {} does not carry parameters for the {} engine",
                    path.display(),
                    engine_name
                ),
                "regenerate the file with Kizzasi::save_weights for this model_type",
            )),
        }
    }

    /// Load a `.safetensors` checkpoint into a `kizzasi-model` architecture.
    fn load_safetensors(&mut self, path: &Path) -> KizzasiResult<()> {
        let arch = match self {
            Self::Architecture(arch) => arch,
            Self::Selective(_) => {
                return Err(KizzasiError::model_not_ready(
                    format!(
                        "the SelectiveSSM engine cannot read the safetensors checkpoint {}",
                        path.display()
                    ),
                    "save its parameters with Kizzasi::save_weights (JSON weight file), or \
                     select ModelType::Mamba / S4 / Rwkv for safetensors interop",
                ))
            }
        };

        let loader = kizzasi_model::loader::ModelLoader::new(path).map_err(|e| {
            model_err(
                &format!("failed to open safetensors file {}", path.display()),
                e,
            )
        })?;
        let tensors = loader
            .load_all()
            .map_err(|e| model_err("failed to read safetensors tensors", e))?;
        let map: HashMap<String, Vec<f32>> = tensors
            .into_iter()
            .map(|(name, array)| (name, array.iter().copied().collect()))
            .collect();
        apply_tensor_map(arch, &map, path)
    }
}

fn check_dim(name: &str, from_file: usize, from_config: usize) -> KizzasiResult<()> {
    if from_file == from_config {
        return Ok(());
    }
    Err(KizzasiError::dimension_mismatch(
        from_config,
        from_file,
        format!("weight file {name} does not match the configuration"),
    ))
}

/// Apply a flat weight map, refusing a load that matched nothing.
///
/// `kizzasi-model`'s loaders deliberately allow partial loads, which means a
/// map whose names match nothing at all returns `Ok` while leaving the model
/// fully random. That is exactly the silent-random-init failure this facade
/// must never ship, so an empty application is turned into an error here.
fn apply_tensor_map(
    arch: &mut ArchitectureBackend,
    map: &HashMap<String, Vec<f32>>,
    path: &Path,
) -> KizzasiResult<()> {
    let applied = arch.model.load_weights_map(map).map_err(|e| {
        model_err(
            &format!("failed to load weights from {}", path.display()),
            e,
        )
    })?;
    if applied == 0 {
        return Err(KizzasiError::model_not_ready(
            format!(
                "no tensor in {} matched a parameter of the selected model \
                 ({} tensors present); the model would have stayed randomly initialised",
                path.display(),
                map.len()
            ),
            "check that the checkpoint was produced for this architecture and that the \
             tensor names use the kizzasi-model convention",
        ));
    }
    Ok(())
}

fn restore_readout(
    arch: &mut ArchitectureBackend,
    readout: Option<&[f32]>,
    config: &KizzasiConfig,
) -> KizzasiResult<()> {
    let native_out = config.get_input_dim();
    let output_dim = config.get_output_dim();
    match (readout, arch.readout.is_some()) {
        (Some(values), true) => {
            let expected = native_out * output_dim;
            if values.len() != expected {
                return Err(KizzasiError::dimension_mismatch(
                    expected,
                    values.len(),
                    "readout projection in the weight file has the wrong length",
                ));
            }
            let projection = Array2::from_shape_vec((native_out, output_dim), values.to_vec())
                .map_err(|e| {
                    KizzasiError::inference(format!("failed to rebuild readout projection: {e}"))
                })?;
            arch.readout = Some(projection);
            Ok(())
        }
        (None, false) => Ok(()),
        (Some(_), false) => Err(KizzasiError::model_not_ready(
            "weight file carries a readout projection but this configuration needs none \
             (input_dim == output_dim)",
            "load the weights with the configuration they were saved from",
        )),
        (None, true) => Err(KizzasiError::model_not_ready(
            "weight file carries no readout projection but this configuration needs one \
             (input_dim != output_dim)",
            "load the weights with the configuration they were saved from",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_for(model_type: ModelType) -> KizzasiConfig {
        KizzasiConfig::new()
            .model_type(model_type)
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(32)
            .state_dim(8)
            .num_layers(2)
    }

    #[test]
    fn selective_backend_is_used_for_mamba2() {
        let backend = Backend::from_config(&config_for(ModelType::Mamba2)).unwrap();
        assert!(backend.selective().is_some());
    }

    #[test]
    fn architecture_backends_are_used_for_other_model_types() {
        for model_type in [ModelType::Mamba, ModelType::S4, ModelType::Rwkv] {
            let backend = Backend::from_config(&config_for(model_type)).unwrap();
            assert!(
                backend.selective().is_none(),
                "{model_type:?} must not fall back to SelectiveSSM"
            );
        }
    }

    #[test]
    fn default_num_heads_divides_hidden_dim() {
        assert_eq!(default_num_heads(32, 8), 8);
        assert_eq!(default_num_heads(6, 8), 6);
        assert_eq!(default_num_heads(7, 8), 7);
        assert_eq!(default_num_heads(1, 8), 1);
    }

    #[test]
    fn unsupported_options_are_rejected_not_ignored() {
        let config = config_for(ModelType::Mamba2).expansion_factor(4);
        assert!(Backend::from_config(&config).is_err());
    }

    #[test]
    fn architecture_backend_cannot_be_cloned() {
        let backend = Backend::from_config(&config_for(ModelType::Rwkv)).unwrap();
        assert!(backend.try_clone().is_err());
    }
}
