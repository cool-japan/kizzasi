//! AdamW with inspectable, persistable optimizer state
//!
//! [`KizzasiAdamW`] reproduces `candle_nn::AdamW` exactly — same decoupled
//! weight decay, same bias correction, same epsilon placement — but keys the
//! first/second moment estimates by parameter **name** and exposes them, so a
//! training run can be resumed without restarting Adam from zero.
//!
//! candle's own `AdamW` stores its moments in private fields with no accessor,
//! which is why a checkpoint written from it can only ever contain weights:
//! resuming would silently discard the moment estimates and produce the classic
//! post-resume loss spike.

use crate::error::{CoreError, CoreResult};
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{ParamsAdamW, VarMap};
use std::collections::HashMap;
use std::path::Path;

/// safetensors key holding the AdamW step counter.
const STEP_KEY: &str = "__kizzasi_adamw_step_t";
/// safetensors key suffix for the first-moment (exponential average) buffer.
const FIRST_MOMENT_SUFFIX: &str = ".exp_avg";
/// safetensors key suffix for the second-moment (squared average) buffer.
const SECOND_MOMENT_SUFFIX: &str = ".exp_avg_sq";

/// Per-parameter AdamW state.
#[derive(Debug)]
struct AdamWSlot {
    /// Parameter name, as registered in the [`VarMap`]
    name: String,
    /// The parameter itself
    var: Var,
    /// Exponential moving average of the gradient
    first_moment: Var,
    /// Exponential moving average of the squared gradient
    second_moment: Var,
}

/// AdamW optimizer whose moment estimates can be saved and restored.
///
/// The update rule is byte-for-byte the one in `candle_nn::AdamW`:
///
/// ```text
/// m ← β₁·m + (1-β₁)·g
/// v ← β₂·v + (1-β₂)·g²
/// θ ← θ·(1 - lr·λ) - lr · (m/(1-β₁ᵗ)) / (sqrt(v/(1-β₂ᵗ)) + ε)
/// ```
///
/// Integer-typed variables are skipped, matching candle.
#[derive(Debug)]
pub struct KizzasiAdamW {
    slots: Vec<AdamWSlot>,
    step_t: usize,
    params: ParamsAdamW,
}

impl KizzasiAdamW {
    /// Build an optimizer over every float parameter registered in `varmap`.
    ///
    /// The [`VarMap`] is what supplies the stable parameter names that make
    /// optimizer-state persistence meaningful.
    pub fn from_varmap(varmap: &VarMap, params: ParamsAdamW) -> CoreResult<Self> {
        let data = varmap
            .data()
            .lock()
            .map_err(|_| CoreError::TrainingError("VarMap mutex poisoned".to_string()))?;

        let mut named: Vec<(String, Var)> = data
            .iter()
            .filter(|(_, var)| var.dtype().is_float())
            .map(|(name, var)| (name.clone(), var.clone()))
            .collect();
        // `VarMap` is backed by a HashMap, so fix a deterministic order.
        named.sort_by(|a, b| a.0.cmp(&b.0));
        drop(data);

        let mut slots = Vec::with_capacity(named.len());
        for (name, var) in named {
            let first_moment = Var::zeros(var.shape(), var.dtype(), var.device()).map_err(|e| {
                CoreError::TrainingError(format!(
                    "Failed to allocate first moment for {}: {}",
                    name, e
                ))
            })?;
            let second_moment =
                Var::zeros(var.shape(), var.dtype(), var.device()).map_err(|e| {
                    CoreError::TrainingError(format!(
                        "Failed to allocate second moment for {}: {}",
                        name, e
                    ))
                })?;
            slots.push(AdamWSlot {
                name,
                var,
                first_moment,
                second_moment,
            });
        }

        Ok(Self {
            slots,
            step_t: 0,
            params,
        })
    }

    /// Current learning rate
    pub fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    /// Set the learning rate (used by the LR schedulers)
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    /// Hyperparameters in use
    pub fn params(&self) -> &ParamsAdamW {
        &self.params
    }

    /// Number of update steps applied so far (drives the bias correction)
    pub fn step_count(&self) -> usize {
        self.step_t
    }

    /// Number of parameters under management
    pub fn num_params(&self) -> usize {
        self.slots.len()
    }

    /// Borrow the first-moment buffer of a named parameter
    pub fn first_moment(&self, name: &str) -> Option<&Var> {
        self.slots
            .iter()
            .find(|s| s.name == name)
            .map(|s| &s.first_moment)
    }

    /// Borrow the second-moment buffer of a named parameter
    pub fn second_moment(&self, name: &str) -> Option<&Var> {
        self.slots
            .iter()
            .find(|s| s.name == name)
            .map(|s| &s.second_moment)
    }

    /// Device the optimizer state lives on
    fn device(&self) -> Device {
        self.slots
            .first()
            .map(|s| s.var.device().clone())
            .unwrap_or(Device::Cpu)
    }

    /// Apply one AdamW update using the gradients in `grads`
    pub fn step(&mut self, grads: &GradStore) -> CoreResult<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));

        for slot in self.slots.iter() {
            let theta = &slot.var;
            let m = &slot.first_moment;
            let v = &slot.second_moment;

            let g = match grads.get(theta) {
                Some(g) => g,
                None => continue,
            };

            let update = || -> candle_core::Result<()> {
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
                Ok(())
            };

            update().map_err(|e| {
                CoreError::TrainingError(format!("AdamW update failed for {}: {}", slot.name, e))
            })?;
        }

        Ok(())
    }

    /// Write the moment estimates and step counter to a safetensors file.
    ///
    /// Buffers are stored as `<param>.exp_avg` / `<param>.exp_avg_sq`, keyed by
    /// the parameter names from the [`VarMap`], so a restore is independent of
    /// hash iteration order.
    pub fn save_state<P: AsRef<Path>>(&self, path: P) -> CoreResult<()> {
        let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(self.slots.len() * 2 + 1);

        for slot in self.slots.iter() {
            tensors.insert(
                format!("{}{}", slot.name, FIRST_MOMENT_SUFFIX),
                slot.first_moment.as_tensor().clone(),
            );
            tensors.insert(
                format!("{}{}", slot.name, SECOND_MOMENT_SUFFIX),
                slot.second_moment.as_tensor().clone(),
            );
        }

        let step = Tensor::new(&[self.step_t as u32], &self.device()).map_err(|e| {
            CoreError::TrainingError(format!("Failed to encode optimizer step: {}", e))
        })?;
        tensors.insert(STEP_KEY.to_string(), step);

        candle_core::safetensors::save(&tensors, path)
            .map_err(|e| CoreError::TrainingError(format!("Failed to save optimizer state: {}", e)))
    }

    /// Restore the moment estimates and step counter from a safetensors file.
    ///
    /// # Errors
    /// Returns [`CoreError::TrainingError`] when the file is unreadable, when a
    /// parameter's buffers are missing, or when a buffer's shape/dtype does not
    /// match the live parameter — a partially restored optimizer is worse than
    /// a reported failure.
    pub fn load_state<P: AsRef<Path>>(&mut self, path: P) -> CoreResult<()> {
        let device = self.device();
        let tensors = candle_core::safetensors::load(path, &device).map_err(|e| {
            CoreError::TrainingError(format!("Failed to load optimizer state: {}", e))
        })?;

        let step = tensors.get(STEP_KEY).ok_or_else(|| {
            CoreError::TrainingError(format!(
                "Optimizer state file is missing the '{}' entry",
                STEP_KEY
            ))
        })?;
        let step_value = step
            .to_dtype(DType::U32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<u32>())
            .map_err(|e| {
                CoreError::TrainingError(format!("Failed to decode optimizer step: {}", e))
            })?;
        self.step_t = step_value.first().copied().unwrap_or(0) as usize;

        for slot in self.slots.iter() {
            let m_key = format!("{}{}", slot.name, FIRST_MOMENT_SUFFIX);
            let v_key = format!("{}{}", slot.name, SECOND_MOMENT_SUFFIX);

            let m = tensors.get(&m_key).ok_or_else(|| {
                CoreError::TrainingError(format!("Optimizer state is missing '{}'", m_key))
            })?;
            let v = tensors.get(&v_key).ok_or_else(|| {
                CoreError::TrainingError(format!("Optimizer state is missing '{}'", v_key))
            })?;

            let m = m.to_dtype(slot.var.dtype()).map_err(|e| {
                CoreError::TrainingError(format!("Failed to cast '{}': {}", m_key, e))
            })?;
            let v = v.to_dtype(slot.var.dtype()).map_err(|e| {
                CoreError::TrainingError(format!("Failed to cast '{}': {}", v_key, e))
            })?;

            // `Var::set` rejects a shape mismatch, so a checkpoint from a
            // differently sized model fails loudly instead of half-loading.
            slot.first_moment.set(&m).map_err(|e| {
                CoreError::TrainingError(format!("Failed to restore '{}': {}", m_key, e))
            })?;
            slot.second_moment.set(&v).map_err(|e| {
                CoreError::TrainingError(format!("Failed to restore '{}': {}", v_key, e))
            })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{AdamW, Optimizer, VarBuilder};

    /// Build a VarMap with two small float parameters plus a matching optimizer.
    fn build_varmap(seed: f64) -> CoreResult<VarMap> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let _ = vb
            .get_with_hints((2, 3), "w", candle_nn::init::Init::Const(seed))
            .map_err(|e| CoreError::TrainingError(format!("{e}")))?;
        let _ = vb
            .get_with_hints(4, "b", candle_nn::init::Init::Const(seed * 0.5))
            .map_err(|e| CoreError::TrainingError(format!("{e}")))?;
        Ok(varmap)
    }

    /// Deterministic pseudo-gradients keyed by the given varmap's vars.
    fn make_grads(varmap: &VarMap, step: usize) -> GradStore {
        let mut grads = GradStore::default();
        let data = varmap.data().lock().unwrap();
        let mut names: Vec<&String> = data.keys().collect();
        names.sort();
        for (i, name) in names.iter().enumerate() {
            let var = &data[*name];
            let elems = var.elem_count();
            let values: Vec<f32> = (0..elems)
                .map(|j| ((i + 1) as f32) * 0.01 * ((j as f32) + 1.0) - 0.005 * (step as f32))
                .collect();
            let g = Tensor::from_vec(values, var.shape(), var.device()).unwrap();
            grads.insert(var, g);
        }
        grads
    }

    fn snapshot(varmap: &VarMap) -> Vec<(String, Vec<f32>)> {
        let data = varmap.data().lock().unwrap();
        let mut out: Vec<(String, Vec<f32>)> = data
            .iter()
            .map(|(name, var)| {
                (
                    name.clone(),
                    var.as_tensor()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                )
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    #[test]
    fn test_matches_candle_adamw_step_for_step() {
        // Guard against silent drift from candle's update rule: the two
        // optimizers must stay in lockstep over multiple steps, including the
        // bias-correction warm-up where the step counter matters most.
        let params = ParamsAdamW {
            lr: 1e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-2,
        };

        let ours_map = build_varmap(0.25).unwrap();
        let theirs_map = build_varmap(0.25).unwrap();

        let mut ours = KizzasiAdamW::from_varmap(&ours_map, params.clone()).unwrap();
        let mut theirs = AdamW::new(theirs_map.all_vars(), params).unwrap();

        assert_eq!(ours.num_params(), 2);

        for step in 0..5 {
            ours.step(&make_grads(&ours_map, step)).unwrap();
            theirs.step(&make_grads(&theirs_map, step)).unwrap();

            let a = snapshot(&ours_map);
            let b = snapshot(&theirs_map);
            assert_eq!(a.len(), b.len());
            for ((na, va), (nb, vb_)) in a.iter().zip(b.iter()) {
                assert_eq!(na, nb);
                for (i, (x, y)) in va.iter().zip(vb_.iter()).enumerate() {
                    assert!(
                        (x - y).abs() < 1e-7,
                        "step {step}: {na}[{i}] diverged from candle AdamW: {x} vs {y}"
                    );
                }
            }
        }
        assert_eq!(ours.step_count(), 5);
    }

    #[test]
    fn test_moments_round_trip_through_disk() {
        let params = ParamsAdamW::default();
        let varmap = build_varmap(0.1).unwrap();
        let mut opt = KizzasiAdamW::from_varmap(&varmap, params.clone()).unwrap();

        for step in 0..3 {
            opt.step(&make_grads(&varmap, step)).unwrap();
        }

        let before_m = opt
            .first_moment("w")
            .unwrap()
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            before_m.iter().any(|v| v.abs() > 1e-9),
            "test precondition: the first moment must be non-zero"
        );

        let path = std::env::temp_dir().join("kizzasi_optimizer_state_roundtrip.safetensors");
        opt.save_state(&path).unwrap();

        // A fresh optimizer starts from zeroed moments and step 0.
        let mut restored = KizzasiAdamW::from_varmap(&varmap, params).unwrap();
        assert_eq!(restored.step_count(), 0);
        restored.load_state(&path).unwrap();

        assert_eq!(restored.step_count(), 3);
        let after_m = restored
            .first_moment("w")
            .unwrap()
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, (a, b)) in before_m.iter().zip(after_m.iter()).enumerate() {
            assert!((a - b).abs() < 1e-9, "first moment[{i}]: {a} vs {b}");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_state_rejects_a_missing_file() {
        let varmap = build_varmap(0.1).unwrap();
        let mut opt = KizzasiAdamW::from_varmap(&varmap, ParamsAdamW::default()).unwrap();
        let missing =
            std::env::temp_dir().join("kizzasi_optimizer_state_does_not_exist.safetensors");
        let _ = std::fs::remove_file(&missing);
        assert!(opt.load_state(&missing).is_err());
    }

    #[test]
    fn test_set_learning_rate_is_honoured() {
        let varmap = build_varmap(0.5).unwrap();
        let mut opt = KizzasiAdamW::from_varmap(&varmap, ParamsAdamW::default()).unwrap();
        opt.set_learning_rate(0.0);
        let before = snapshot(&varmap);

        opt.step(&make_grads(&varmap, 0)).unwrap();
        let after = snapshot(&varmap);

        // A zero learning rate still applies decoupled weight decay of
        // `lr * weight_decay == 0`, so the parameters must be unchanged.
        for ((_, a), (_, b)) in before.iter().zip(after.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-9, "lr=0 changed a parameter: {x} vs {y}");
            }
        }
    }
}
