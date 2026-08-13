//! S5: Simplified State Space Model
//!
//! S5 simplifies S4 by using a more efficient parameterization of the state space
//! while maintaining competitive performance. Key simplifications include:
//!
//! - **Simplified initialization**: Easier parameter initialization
//! - **Faster computation**: Reduced computational overhead
//! - **Better numerical stability**: Improved gradient flow
//! - **Diagonal state matrix**: Like S4D, but with optimized discretization
//!
//! # Architecture
//!
//! ```text
//! Input → [Linear] → [SSM Block] → [Activation] → [LayerNorm] → Output
//!                         ↓
//!                     [State]
//! ```
//!
//! # SSM Formulation
//!
//! Continuous-time:
//! ```text
//! h'(t) = Ah(t) + Bx(t)
//! y(t) = Ch(t)
//! ```
//!
//! Where A is diagonal and initialized more simply than S4.
//!
//! # References
//!
//! - S5 paper: <https://arxiv.org/abs/2208.04933>
//! - Efficiently Modeling Long Sequences with Structured State Spaces

use crate::error::{ModelError, ModelResult};
use crate::{AutoregressiveModel, ModelType};
use kizzasi_core::{gelu, CoreResult, HiddenState, LayerNorm, NormType, SignalPredictor};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rng, RngExt};

#[allow(unused_imports)]
use tracing::{debug, instrument, trace};

/// Configuration for S5 model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct S5Config {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// State dimension (typically 64-256)
    pub state_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Discretization step size (Δt)
    pub dt: f32,
    /// Reserved for a future chunked parallel-scan implementation.
    ///
    /// `S5Layer::forward` (used by [`S5::step`]) is a single-step
    /// sequential recurrence — the O(1)-per-token path needed for
    /// autoregressive inference — so there is currently no batched/sequence
    /// forward pass for this field to chunk. It is validated as non-zero (a
    /// zero block size can never denote a real chunk) but otherwise has no
    /// effect yet.
    pub block_size: usize,
}

impl S5Config {
    /// Create default S5 configuration
    pub fn new(input_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            state_dim: 64,
            num_layers,
            dt: 0.001,
            block_size: 64,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> ModelResult<()> {
        if self.hidden_dim == 0 {
            return Err(ModelError::invalid_config("hidden_dim must be > 0"));
        }
        if self.state_dim == 0 {
            return Err(ModelError::invalid_config("state_dim must be > 0"));
        }
        if self.num_layers == 0 {
            return Err(ModelError::invalid_config("num_layers must be > 0"));
        }
        if self.dt <= 0.0 {
            return Err(ModelError::invalid_config("dt must be > 0"));
        }
        if self.block_size == 0 {
            return Err(ModelError::invalid_config("block_size must be > 0"));
        }
        Ok(())
    }
}

/// S5 SSM block with diagonal state matrix
#[allow(dead_code)]
struct S5Block {
    /// Diagonal of A matrix (log-space for stability)
    log_a: Array1<f32>,
    /// B matrix [state_dim, hidden_dim]
    b_matrix: Array2<f32>,
    /// C matrix [hidden_dim, state_dim]
    c_matrix: Array2<f32>,
    /// D skip connection [hidden_dim]
    d_vec: Array1<f32>,
    /// Discretization step
    dt: f32,
    /// Discretized A diagonal
    a_bar: Array1<f32>,
    /// Discretized B matrix
    b_bar: Array2<f32>,
    /// Current state [state_dim]
    state: Array1<f32>,
}

impl S5Block {
    /// Create new S5 block with simplified initialization
    fn new(hidden_dim: usize, state_dim: usize, dt: f32) -> Self {
        let mut rng = rng();

        // Initialize log_a as HiPPO-LegS diagonal magnitudes: log|(-(2n+1)/2)|
        // A[n] = -(2n+1)/2, storing log of absolute value so A[n] = -exp(log_a[n]) < 0
        let log_a = Array1::from_shape_fn(state_dim, |i| ((2 * i + 1) as f32 / 2.0).ln());

        // Initialize B and C with random values
        let scale_b = (2.0 / (state_dim + hidden_dim) as f32).sqrt();
        let b_matrix = Array2::from_shape_fn((state_dim, hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale_b
        });

        let scale_c = (2.0 / (hidden_dim + state_dim) as f32).sqrt();
        let c_matrix = Array2::from_shape_fn((hidden_dim, state_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale_c
        });

        // Initialize D (skip connection) to small values
        let d_vec = Array1::from_shape_fn(hidden_dim, |_| rng.random::<f32>() * 0.01);

        // Discretize using zero-order hold (ZOH)
        // A[i] = -exp(log_a[i]) < 0, so a_bar[i] = exp(dt * A[i]) in (0, 1)
        // B̄[i,:] = B[i,:] * (1 - a_bar[i]) / (-A[i])  — proper ZOH B scale
        let mut a_bar = Array1::zeros(state_dim);
        let mut b_bar = Array2::zeros(b_matrix.raw_dim());
        for i in 0..state_dim {
            let a_i = -log_a[i].exp(); // negative: a_i < 0
            a_bar[i] = (dt * a_i).exp(); // 0 < a_bar < 1
            let scale = (1.0 - a_bar[i]) / (-a_i);
            for j in 0..hidden_dim {
                b_bar[[i, j]] = b_matrix[[i, j]] * scale;
            }
        }

        let state = Array1::zeros(state_dim);

        Self {
            log_a,
            b_matrix,
            c_matrix,
            d_vec,
            dt,
            a_bar,
            b_bar,
            state,
        }
    }

    /// Forward pass through S5 block
    #[instrument(skip(self, x))]
    fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // Update state: h[t] = A̅·h[t-1] + B̅·x[t]
        self.state = &self.state * &self.a_bar + self.b_bar.dot(x);

        // Compute output: y[t] = C·h[t] + D·x[t]
        let y = self.c_matrix.dot(&self.state) + &self.d_vec * x;

        Ok(y)
    }

    /// Reset the state
    fn reset(&mut self) {
        self.state.fill(0.0);
    }

    /// Recompute the zero-order-hold discretization from `log_a`, `b_matrix`
    /// and `dt`.
    ///
    /// Must be called after any of those three are replaced (e.g. by weight
    /// loading); otherwise the block would keep stepping with the discretized
    /// matrices derived from the previous parameters.
    fn rediscretize(&mut self) {
        let state_dim = self.log_a.len();
        let hidden_dim = self.b_matrix.shape().get(1).copied().unwrap_or(0);
        let mut a_bar = Array1::zeros(state_dim);
        let mut b_bar = Array2::zeros(self.b_matrix.raw_dim());
        for i in 0..state_dim {
            let a_i = -self.log_a[i].exp();
            a_bar[i] = (self.dt * a_i).exp();
            let scale = (1.0 - a_bar[i]) / (-a_i);
            for j in 0..hidden_dim {
                b_bar[[i, j]] = self.b_matrix[[i, j]] * scale;
            }
        }
        self.a_bar = a_bar;
        self.b_bar = b_bar;
    }
}

/// S5 layer with SSM block, activation, and normalization
struct S5Layer {
    /// Input projection
    input_proj: Array2<f32>,
    /// S5 SSM block
    s5_block: S5Block,
    /// Layer normalization
    layer_norm: LayerNorm,
    /// Output projection
    output_proj: Array2<f32>,
}

impl S5Layer {
    /// Create a new S5 layer
    fn new(config: &S5Config) -> ModelResult<Self> {
        let mut rng = rng();

        // Input projection
        let scale = (2.0 / (config.input_dim + config.hidden_dim) as f32).sqrt();
        let input_proj = Array2::from_shape_fn((config.input_dim, config.hidden_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        // S5 block
        let s5_block = S5Block::new(config.hidden_dim, config.state_dim, config.dt);

        // Layer normalization
        let layer_norm = LayerNorm::new(config.hidden_dim, NormType::RMSNorm);

        // Output projection
        let scale = (2.0 / (config.hidden_dim + config.input_dim) as f32).sqrt();
        let output_proj = Array2::from_shape_fn((config.hidden_dim, config.input_dim), |_| {
            (rng.random::<f32>() - 0.5) * 2.0 * scale
        });

        Ok(Self {
            input_proj,
            s5_block,
            layer_norm,
            output_proj,
        })
    }

    /// Forward pass
    fn forward(&mut self, x: &Array1<f32>) -> CoreResult<Array1<f32>> {
        // Project input
        let hidden = x.dot(&self.input_proj);

        // S5 SSM block
        let ssm_out = self.s5_block.forward(&hidden)?;

        // Activation
        let activated = gelu(&ssm_out);

        // Layer norm
        let normed = self.layer_norm.forward(&activated);

        // Output projection with residual
        let output = normed.dot(&self.output_proj) + x;

        Ok(output)
    }

    /// Reset layer state
    fn reset(&mut self) {
        self.s5_block.reset();
    }
}

/// S5 model with multiple layers
pub struct S5 {
    config: S5Config,
    layers: Vec<S5Layer>,
}

impl S5 {
    /// Create a new S5 model
    #[instrument(skip(config), fields(input_dim = config.input_dim, hidden_dim = config.hidden_dim, num_layers = config.num_layers))]
    pub fn new(config: S5Config) -> ModelResult<Self> {
        debug!("Creating new S5 model");
        config.validate()?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            trace!("Initializing S5 layer {}", layer_idx);
            layers.push(S5Layer::new(&config)?);
        }
        debug!("Initialized {} S5 layers", layers.len());

        debug!("S5 model created successfully");
        Ok(Self { config, layers })
    }

    /// Get configuration
    pub fn config(&self) -> &S5Config {
        &self.config
    }

    /// Load weights from a JSON file holding a `HashMap<String, Vec<f32>>`.
    ///
    /// Only keys present in the file are applied; missing keys leave the current
    /// randomly-initialized values in place (graceful partial loading).
    pub fn load_weights_json<P: AsRef<std::path::Path>>(&mut self, path: P) -> ModelResult<()> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| {
            ModelError::load_error("s5 load_weights", format!("failed to open file: {e}"))
        })?;
        let weights: std::collections::HashMap<String, Vec<f32>> = serde_json::from_reader(file)
            .map_err(|e| {
                ModelError::load_error(
                    "s5 load_weights",
                    format!("JSON deserialization failed: {e}"),
                )
            })?;
        self.load_weights_map(&weights).map(|_| ())
    }

    /// Load weights from an in-memory `name → flat f32 values` map.
    ///
    /// # Expected keys
    ///
    /// For each layer `i` in `0..num_layers`:
    /// - `layers.{i}.input_proj`: `[input_dim, hidden_dim]`
    /// - `layers.{i}.output_proj`: `[hidden_dim, input_dim]`
    /// - `layers.{i}.s5_block.log_a`: `[state_dim]`
    /// - `layers.{i}.s5_block.b_matrix`: `[state_dim, hidden_dim]`
    /// - `layers.{i}.s5_block.c_matrix`: `[hidden_dim, state_dim]`
    /// - `layers.{i}.s5_block.d_vec`: `[hidden_dim]`
    ///
    /// Replacing `log_a` or `b_matrix` invalidates the zero-order-hold
    /// discretization, so it is recomputed for every layer that was touched.
    pub fn load_weights_map(
        &mut self,
        weights: &std::collections::HashMap<String, Vec<f32>>,
    ) -> ModelResult<usize> {
        // Number of tensors actually applied. The caller needs this to tell a
        // genuine partial load from a weight map whose names match nothing at
        // all — the latter would otherwise leave the model randomly
        // initialised while reporting success.
        let applied = std::cell::Cell::new(0usize);
        let load_array2 = |map: &std::collections::HashMap<String, Vec<f32>>,
                           key: &str,
                           rows: usize,
                           cols: usize|
         -> ModelResult<Option<Array2<f32>>> {
            if let Some(data) = map.get(key) {
                if data.len() != rows * cols {
                    return Err(ModelError::load_error(
                        "s5 load_weights",
                        format!(
                            "shape mismatch for '{}': expected {}×{}={} but got {}",
                            key,
                            rows,
                            cols,
                            rows * cols,
                            data.len()
                        ),
                    ));
                }
                let arr = Array2::from_shape_vec((rows, cols), data.clone()).map_err(|e| {
                    ModelError::load_error(
                        "s5 load_weights",
                        format!("failed to reshape '{}': {e}", key),
                    )
                })?;
                applied.set(applied.get() + 1);
                Ok(Some(arr))
            } else {
                Ok(None)
            }
        };

        let load_array1 = |map: &std::collections::HashMap<String, Vec<f32>>,
                           key: &str,
                           expected_len: usize|
         -> ModelResult<Option<Array1<f32>>> {
            if let Some(data) = map.get(key) {
                if data.len() != expected_len {
                    return Err(ModelError::load_error(
                        "s5 load_weights",
                        format!(
                            "shape mismatch for '{}': expected {} but got {}",
                            key,
                            expected_len,
                            data.len()
                        ),
                    ));
                }
                applied.set(applied.get() + 1);
                Ok(Some(Array1::from_vec(data.clone())))
            } else {
                Ok(None)
            }
        };

        let input_dim = self.config.input_dim;
        let hidden = self.config.hidden_dim;
        let state = self.config.state_dim;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);
            let bp = format!("{}.s5_block", prefix);

            if let Some(arr) = load_array2(
                weights,
                &format!("{}.input_proj", prefix),
                input_dim,
                hidden,
            )? {
                layer.input_proj = arr;
            }
            if let Some(arr) = load_array2(
                weights,
                &format!("{}.output_proj", prefix),
                hidden,
                input_dim,
            )? {
                layer.output_proj = arr;
            }

            let mut discretization_stale = false;
            if let Some(arr) = load_array1(weights, &format!("{}.log_a", bp), state)? {
                layer.s5_block.log_a = arr;
                discretization_stale = true;
            }
            if let Some(arr) = load_array2(weights, &format!("{}.b_matrix", bp), state, hidden)? {
                layer.s5_block.b_matrix = arr;
                discretization_stale = true;
            }
            if let Some(arr) = load_array2(weights, &format!("{}.c_matrix", bp), hidden, state)? {
                layer.s5_block.c_matrix = arr;
            }
            if let Some(arr) = load_array1(weights, &format!("{}.d_vec", bp), hidden)? {
                layer.s5_block.d_vec = arr;
            }

            if discretization_stale {
                layer.s5_block.rediscretize();
            }
        }

        Ok(applied.get())
    }

    /// Serialize weights to an in-memory `name → flat f32 values` map.
    ///
    /// Covers exactly the keys [`Self::load_weights_map`] reads back — see
    /// its doc for the full key list — so `save_weights_map` followed by
    /// `load_weights_map` round-trips every parameter this model exposes a
    /// way to read. `layer_norm`'s gain/bias are the one exception: this
    /// crate's `kizzasi_core::LayerNorm` exposes `set_gamma`/`set_beta`
    /// setters but no getters, so there is no way to read the *current*
    /// normalization parameters back out to serialize them (this mirrors
    /// `load_weights_map`, which correspondingly never touches
    /// `layer_norm` either — the round trip is symmetric, not silently
    /// lossy relative to what loading itself already restores).
    pub fn save_weights_map(&self) -> std::collections::HashMap<String, Vec<f32>> {
        let mut weights = std::collections::HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);
            let bp = format!("{}.s5_block", prefix);

            weights.insert(
                format!("{}.input_proj", prefix),
                layer.input_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.output_proj", prefix),
                layer.output_proj.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.log_a", bp),
                layer.s5_block.log_a.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.b_matrix", bp),
                layer.s5_block.b_matrix.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.c_matrix", bp),
                layer.s5_block.c_matrix.iter().copied().collect(),
            );
            weights.insert(
                format!("{}.d_vec", bp),
                layer.s5_block.d_vec.iter().copied().collect(),
            );
        }
        weights
    }

    /// Save weights to a JSON file holding a `HashMap<String, Vec<f32>>`
    /// (see [`Self::save_weights_map`] for exactly what is covered).
    pub fn save_weights_json<P: AsRef<std::path::Path>>(&self, path: P) -> ModelResult<()> {
        let weights = self.save_weights_map();
        let file = std::fs::File::create(path.as_ref()).map_err(|e| {
            ModelError::load_error("s5 save_weights", format!("failed to create file: {e}"))
        })?;
        serde_json::to_writer(file, &weights).map_err(|e| {
            ModelError::load_error("s5 save_weights", format!("JSON serialization failed: {e}"))
        })
    }
}

impl SignalPredictor for S5 {
    #[instrument(skip(self, input))]
    fn step(&mut self, input: &Array1<f32>) -> CoreResult<Array1<f32>> {
        crate::check_input_dim(input, self.config.input_dim)?;

        let mut x = input.clone();

        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }

        Ok(x)
    }

    #[instrument(skip(self))]
    fn reset(&mut self) {
        debug!("Resetting S5 model state");
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    fn context_window(&self) -> usize {
        // SSMs have theoretically infinite context via recurrence
        usize::MAX
    }
}

impl AutoregressiveModel for S5 {
    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    fn state_dim(&self) -> usize {
        self.config.state_dim
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn model_type(&self) -> ModelType {
        ModelType::S4 // S5 is a variant of S4
    }

    fn get_states(&self) -> Vec<HiddenState> {
        self.layers
            .iter()
            .map(|layer| {
                // S5 uses 1D state, so expand to 2D for HiddenState
                let state_1d = layer.s5_block.state.clone();
                let state_2d = state_1d.insert_axis(scirs2_core::ndarray::Axis(0));
                let mut hidden_state = HiddenState::new(
                    self.config.hidden_dim,
                    state_2d.len_of(scirs2_core::ndarray::Axis(1)),
                );
                hidden_state.update(state_2d);
                hidden_state
            })
            .collect()
    }

    fn set_states(&mut self, states: Vec<HiddenState>) -> ModelResult<()> {
        if states.len() != self.config.num_layers {
            return Err(ModelError::state_count_mismatch(
                "S5",
                self.config.num_layers,
                states.len(),
            ));
        }

        for (layer, state) in self.layers.iter_mut().zip(states.iter()) {
            // Convert from 2D back to 1D
            let state_2d = state.state();
            if state_2d.nrows() > 0 && state_2d.ncols() > 0 {
                layer.s5_block.state = state_2d.row(0).to_owned();
            }
        }

        Ok(())
    }

    fn load_weights_json(&mut self, path: &std::path::Path) -> ModelResult<()> {
        S5::load_weights_json(self, path)
    }

    fn save_weights_json(&self, path: &std::path::Path) -> ModelResult<()> {
        S5::save_weights_json(self, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_s5() -> (S5Config, S5) {
        let mut config = S5Config::new(2, 4, 1);
        config.state_dim = 3;
        let model = S5::new(config.clone()).expect("S5::new");
        (config, model)
    }

    #[test]
    fn test_load_weights_map_applies_projections() {
        let (config, mut model) = small_s5();
        let before = model.layers[0].input_proj.clone();

        let values: Vec<f32> = (0..(config.input_dim * config.hidden_dim))
            .map(|i| i as f32 * 0.125)
            .collect();
        let mut weights = std::collections::HashMap::new();
        weights.insert("layers.0.input_proj".to_string(), values.clone());

        let applied = model.load_weights_map(&weights).expect("load_weights_map");
        assert_eq!(applied, 1, "exactly one tensor should have been applied");
        assert_ne!(model.layers[0].input_proj, before);
        for (i, v) in model.layers[0].input_proj.iter().enumerate() {
            assert!((v - values[i]).abs() < 1e-6, "element {i}");
        }
    }

    #[test]
    fn test_load_weights_map_rediscretizes_ssm() {
        let (config, mut model) = small_s5();
        let a_bar_before = model.layers[0].s5_block.a_bar.clone();
        let b_bar_before = model.layers[0].s5_block.b_bar.clone();

        // A different log_a and B must invalidate the cached ZOH matrices.
        let log_a: Vec<f32> = (0..config.state_dim).map(|i| 1.0 + i as f32).collect();
        let b: Vec<f32> = (0..(config.state_dim * config.hidden_dim))
            .map(|i| 0.5 + i as f32 * 0.25)
            .collect();
        let mut weights = std::collections::HashMap::new();
        weights.insert("layers.0.s5_block.log_a".to_string(), log_a.clone());
        weights.insert("layers.0.s5_block.b_matrix".to_string(), b.clone());

        let applied = model.load_weights_map(&weights).expect("load_weights_map");
        assert_eq!(applied, 2);

        assert_ne!(
            model.layers[0].s5_block.a_bar, a_bar_before,
            "a_bar must be recomputed after log_a changes"
        );
        assert_ne!(
            model.layers[0].s5_block.b_bar, b_bar_before,
            "b_bar must be recomputed after B changes"
        );

        // Check the ZOH formula element-wise against the freshly loaded values.
        let dt = config.dt;
        for i in 0..config.state_dim {
            let a_i = -log_a[i].exp();
            let expected_a = (dt * a_i).exp();
            assert!(
                (model.layers[0].s5_block.a_bar[i] - expected_a).abs() < 1e-6,
                "a_bar[{i}]: {} != {expected_a}",
                model.layers[0].s5_block.a_bar[i]
            );
            let scale = (1.0 - expected_a) / (-a_i);
            for j in 0..config.hidden_dim {
                let expected_b = b[i * config.hidden_dim + j] * scale;
                assert!(
                    (model.layers[0].s5_block.b_bar[[i, j]] - expected_b).abs() < 1e-6,
                    "b_bar[{i},{j}]: {} != {expected_b}",
                    model.layers[0].s5_block.b_bar[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_load_weights_map_reports_unmatched_names() {
        let (_config, mut model) = small_s5();
        let mut weights = std::collections::HashMap::new();
        weights.insert("blocks.0.attn.q_proj".to_string(), vec![0.0f32; 8]);
        let applied = model.load_weights_map(&weights).expect("load_weights_map");
        assert_eq!(
            applied, 0,
            "unknown names must report zero applied tensors, not a silent success"
        );
    }

    #[test]
    fn test_load_weights_map_reports_shape_mismatch() {
        let (_config, mut model) = small_s5();
        let mut weights = std::collections::HashMap::new();
        weights.insert("layers.0.input_proj".to_string(), vec![0.0f32; 3]);
        let err = model
            .load_weights_map(&weights)
            .expect_err("wrong shape must be rejected");
        assert!(
            err.to_string().contains("layers.0.input_proj"),
            "got: {err}"
        );
    }

    #[test]
    fn test_s5_creation() {
        let config = S5Config::new(32, 64, 2);
        let model = S5::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_s5_forward() {
        let config = S5Config::new(32, 64, 2);
        let mut model = S5::new(config).expect("Failed to create S5 model");

        let input = Array1::from_vec(vec![1.0; 32]);
        let output = model.step(&input);
        assert!(output.is_ok());
        assert_eq!(output.expect("Failed to get output").len(), 32);
    }

    #[test]
    fn test_s5_reset() {
        let config = S5Config::new(32, 64, 2);
        let mut model = S5::new(config).expect("Failed to create S5 model");

        let input = Array1::from_vec(vec![1.0; 32]);
        let _output1 = model.step(&input).expect("Failed to get output1");

        model.reset();

        let output2 = model.step(&input).expect("Failed to get output2");
        // After reset, same input should give similar output to first step
        assert_eq!(output2.len(), 32);
    }

    #[test]
    fn test_s5_save_load_json_round_trip_identical_step_output() {
        // Regression test for id103's real residual gap: S5 previously had
        // no `save_weights_json` at all (the `AutoregressiveModel` trait
        // default unconditionally errors), so an S5 model could never be
        // persisted. `load_weights_json`/`create_s5` already worked -- this
        // specifically exercises the save side and a full round trip.
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let uid = COUNTER.fetch_add(1, Ordering::Relaxed);

        let config = S5Config::new(6, 12, 2);
        let mut model_a = S5::new(config.clone()).expect("S5::new (a)");

        // Give the model non-default weights so a round trip that silently
        // fell back to fresh random init would be detectable.
        let input_proj_values: Vec<f32> = (0..(config.input_dim * config.hidden_dim))
            .map(|i| 0.01 * i as f32)
            .collect();
        let mut seed_weights = std::collections::HashMap::new();
        seed_weights.insert("layers.0.input_proj".to_string(), input_proj_values);
        model_a
            .load_weights_map(&seed_weights)
            .expect("seeding model_a with distinct weights should succeed");

        let mut tmp = std::env::temp_dir();
        tmp.push(format!("kizzasi_s5_roundtrip_test_{uid}.json"));

        model_a
            .save_weights_json(&tmp)
            .expect("save_weights_json should succeed");

        let mut model_b = S5::new(config).expect("S5::new (b)");
        model_b
            .load_weights_json(&tmp)
            .expect("load_weights_json should succeed");
        let _ = std::fs::remove_file(&tmp);

        let input = Array1::from_vec(vec![0.2, -0.4, 0.6, -0.1, 0.3, -0.5]);
        let out_a = model_a.step(&input).expect("model_a step");
        let out_b = model_b.step(&input).expect("model_b step");

        assert_eq!(out_a.len(), out_b.len());
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "save->load round trip must reproduce identical step() \
                 output: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_invalid_config() {
        let mut config = S5Config::new(32, 64, 2);
        config.state_dim = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_s5_block_a_bar_stable() {
        // After init, all a_bar values must be in (0, 1) — not > 1
        let block = S5Block::new(8, 16, 0.01);
        for &val in block.a_bar.iter() {
            assert!(
                val > 0.0 && val < 1.0,
                "a_bar = {} not in (0,1) — SSM is unstable",
                val
            );
        }
    }

    #[test]
    fn test_s5_block_state_decays() {
        // Zero-input SSM from nonzero state must not grow
        let mut block = S5Block::new(8, 16, 0.01);
        // Set state to nonzero
        block.state = Array1::ones(16);
        let initial_norm: f32 = block.state.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Run 200 zero-input steps
        let zero_input = Array1::zeros(8);
        for _ in 0..200 {
            let new_state = &block.a_bar * &block.state + block.b_bar.dot(&zero_input);
            block.state = new_state;
        }
        let final_norm: f32 = block.state.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            final_norm <= initial_norm,
            "State grew from {} to {} — unstable!",
            initial_norm,
            final_norm
        );
    }

    #[test]
    fn test_s5_block_hippo_log_a() {
        // log_a[i] must equal ((2i+1)/2).ln()
        let block = S5Block::new(8, 16, 0.01);
        for (i, &val) in block.log_a.iter().enumerate() {
            let expected = ((2 * i + 1) as f32 / 2.0).ln();
            assert!(
                (val - expected).abs() < 1e-5,
                "log_a[{}]={} expected {}",
                i,
                val,
                expected
            );
        }
    }

    #[test]
    fn test_s5_block_a_bar_matches_zoh_formula() {
        // a_bar[i] must match exp(dt * (-exp(log_a[i])))
        let dt = 0.01_f32;
        let block = S5Block::new(8, 16, dt);
        for (i, (&log_a_val, &a_bar_val)) in block.log_a.iter().zip(block.a_bar.iter()).enumerate()
        {
            let a_i = -log_a_val.exp();
            let expected = (dt * a_i).exp();
            assert!(
                (a_bar_val - expected).abs() < 1e-5,
                "a_bar[{}]={} expected {}",
                i,
                a_bar_val,
                expected
            );
        }
    }
}
