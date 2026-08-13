//! Integration-level regression tests for subsystems that previously had
//! zero coverage in `tests/` (only in-module `#[cfg(test)]` unit tests),
//! which is precisely the gap that let several audited bugs survive:
//! `WeightLoadConfig`'s `strict`/`quantize` being stored and never read,
//! `PyTorchConverter::apply_mappings` silently dropping a colliding source
//! tensor, and `parallel_ssm_batch` adding a bare constant instead of a
//! real `D · x_t` skip connection.
//!
//! This file exercises those three subsystems end-to-end, as an external
//! consumer of the crate's public API (matching how a real downstream user
//! would call them), rather than via the crate's own in-module tests.

use candle_core::{DType, Device, Tensor};
use kizzasi_core::*;
use scirs2_core::ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;

// ---------------------------------------------------------------------
// WeightLoader: strict-mode round trip
// ---------------------------------------------------------------------

/// A strict-mode `WeightLoader::load_safetensors` must fail (naming the
/// missing key) when the target `VarMap` needs a variable the saved file
/// does not contain, instead of silently delegating to `VarMap::load` and
/// either panicking deep inside candle or loading a partially-initialised
/// model with no diagnostic at all.
#[test]
fn test_weight_loader_strict_mode_end_to_end() {
    use candle_nn::VarBuilder;

    let device = Device::Cpu;

    // Save a checkpoint containing only "w1" and "w2".
    let save_varmap = candle_nn::VarMap::new();
    let vb_save = VarBuilder::from_varmap(&save_varmap, DType::F32, &device);
    vb_save
        .get_with_hints((4, 4), "w1", candle_nn::init::Init::Const(1.0))
        .unwrap();
    vb_save
        .get_with_hints((2, 2), "w2", candle_nn::init::Init::Const(2.0))
        .unwrap();

    let path = std::env::temp_dir().join("kizzasi_core_integration_strict_checkpoint.safetensors");
    WeightLoader::new(WeightLoadConfig::default())
        .save_safetensors(&path, &save_varmap)
        .unwrap();

    // A "model" that additionally needs "w3", which the file does not have.
    let mut load_varmap = candle_nn::VarMap::new();
    let vb_load = VarBuilder::from_varmap(&load_varmap, DType::F32, &device);
    vb_load
        .get_with_hints((4, 4), "w1", candle_nn::init::Init::Const(0.0))
        .unwrap();
    vb_load
        .get_with_hints((2, 2), "w2", candle_nn::init::Init::Const(0.0))
        .unwrap();
    vb_load
        .get_with_hints((3, 3), "w3", candle_nn::init::Init::Const(0.0))
        .unwrap();

    // Default config is `strict: true`.
    let strict_loader = WeightLoader::new(WeightLoadConfig::default());
    let strict_result = strict_loader.load_safetensors(&path, &mut load_varmap);
    assert!(
        strict_result.is_err(),
        "strict load must fail when the model needs a key the file lacks"
    );
    assert!(
        format!("{}", strict_result.unwrap_err()).contains("w3"),
        "strict load's error must name the missing key"
    );

    // The SAME file, loaded non-strictly, must succeed and leave "w3" at
    // its pre-existing value rather than erroring or corrupting it.
    let mut load_varmap_lenient = candle_nn::VarMap::new();
    let vb_lenient = VarBuilder::from_varmap(&load_varmap_lenient, DType::F32, &device);
    vb_lenient
        .get_with_hints((4, 4), "w1", candle_nn::init::Init::Const(0.0))
        .unwrap();
    vb_lenient
        .get_with_hints((2, 2), "w2", candle_nn::init::Init::Const(0.0))
        .unwrap();
    vb_lenient
        .get_with_hints((3, 3), "w3", candle_nn::init::Init::Const(7.0))
        .unwrap();

    let lenient_config = WeightLoadConfig {
        strict: false,
        ..WeightLoadConfig::default()
    };
    WeightLoader::new(lenient_config)
        .load_safetensors(&path, &mut load_varmap_lenient)
        .expect("non-strict load must succeed despite the missing key");

    let data = load_varmap_lenient.data().lock().unwrap();
    let w1 = data
        .get("w1")
        .unwrap()
        .as_tensor()
        .to_vec2::<f32>()
        .unwrap();
    assert!(w1
        .iter()
        .all(|row| row.iter().all(|&v| (v - 1.0).abs() < 1e-6)));
    let w3 = data
        .get("w3")
        .unwrap()
        .as_tensor()
        .to_vec2::<f32>()
        .unwrap();
    assert!(
        (w3[0][0] - 7.0).abs() < 1e-6,
        "w3 must be left untouched by a non-strict load (no key for it in the file)"
    );
    drop(data);

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------
// PyTorchConverter: mapping-collision end to end
// ---------------------------------------------------------------------

/// A real Mamba `state_dict` contains both `<prefix>.mixer.dt_proj.weight`
/// and `<prefix>.mixer.dt_proj.bias`. `create_mamba_mappings`'s pattern for
/// `dt_proj` is a bare prefix with no `.weight`/`.bias` suffix, so both
/// source tensors match the same target name -- `apply_mappings` must
/// report the collision as an error rather than silently letting one
/// overwrite the other (non-deterministically, depending on `HashMap`
/// iteration order).
#[test]
fn test_pytorch_converter_end_to_end_checkpoint_conversion() {
    let device = Device::Cpu;
    let mut converter = PyTorchConverter::new_cpu();
    converter.create_mamba_mappings();

    let mut state_dict: HashMap<String, Tensor> = HashMap::new();
    state_dict.insert(
        "layers.0.mixer.dt_proj.weight".to_string(),
        Tensor::zeros((16, 8), DType::F32, &device).unwrap(),
    );
    state_dict.insert(
        "layers.0.mixer.dt_proj.bias".to_string(),
        Tensor::zeros(16, DType::F32, &device).unwrap(),
    );

    let result = converter.apply_mappings(state_dict);
    assert!(
        result.is_err(),
        "apply_mappings must surface the dt_proj weight/bias collision as an error"
    );

    // A checkpoint WITHOUT the colliding pair converts cleanly and produces
    // the expected target keys.
    let mut clean_state_dict: HashMap<String, Tensor> = HashMap::new();
    clean_state_dict.insert(
        "layers.0.mixer.in_proj.weight".to_string(),
        Tensor::zeros((32, 16), DType::F32, &device).unwrap(),
    );
    clean_state_dict.insert(
        "layers.0.mixer.A_log".to_string(),
        Tensor::zeros(16, DType::F32, &device).unwrap(),
    );

    let mapped = converter.apply_mappings(clean_state_dict).unwrap();
    assert!(mapped.contains_key("layer_0.in_proj_w"));
    assert!(mapped.contains_key("layer_0.a_log"));

    // detect_architecture on the same converter's own weights, run
    // end-to-end: must be deterministic and report the right architecture.
    let mut weights_for_detection: HashMap<String, Tensor> = HashMap::new();
    weights_for_detection.insert(
        "layers.0.mixer.in_proj.weight".to_string(),
        Tensor::zeros((256, 128), DType::F32, &device).unwrap(),
    );
    let checkpoint = converter
        .detect_architecture(&weights_for_detection)
        .unwrap();
    assert_eq!(checkpoint.architecture, "mamba");
    assert_eq!(checkpoint.d_model, Some(128));
}

// ---------------------------------------------------------------------
// parallel_ssm_batch: end-to-end vs. an independent naive reference
// ---------------------------------------------------------------------

/// Drive `parallel_ssm_batch` as an external caller would (public types
/// only) and check its output against a deliberately independent,
/// straight-line-recurrence reference: `h_t = A_bar * h_{t-1} + B_bar * 1`,
/// `y_t = C . h_t + D * x_t`. This exercises the real `D · x_t` skip
/// connection end to end -- not just that changing `x` at one timestep
/// changes the output there (already covered in-module), but that the
/// FULL batched computation matches manual per-timestep math from scratch.
#[test]
fn test_parallel_ssm_batch_end_to_end_vs_manual_recurrence() {
    let batch_size = 2usize;
    let seq_len = 10usize;
    let state_dim = 5usize;

    let a_bars = Array3::from_shape_fn((batch_size, seq_len, state_dim), |(b, t, d)| {
        0.88 + ((b * 13 + t * 7 + d) % 11) as f32 * 0.005
    });
    let b_bars = Array3::from_shape_fn((batch_size, seq_len, state_dim), |(b, t, d)| {
        0.03 + ((b * 5 + t * 3 + d) % 7) as f32 * 0.002
    });
    let x = Array2::from_shape_fn((batch_size, seq_len), |(b, t)| {
        0.2 * (b as f32 + 1.0) * ((t as f32 * 0.4).sin())
    });
    let c = Array1::from_shape_fn(state_dim, |d| 0.3 + (d as f32) * 0.02);
    let d_skip = 0.15f32;

    let config = ParallelConfig::default();
    let output = parallel_ssm_batch(&a_bars, &b_bars, &x, &c, d_skip, &config).unwrap();
    assert_eq!(output.dim(), (batch_size, seq_len));

    // Independent manual recurrence: h_t = a_bar_t * h_{t-1} + b_bar_t
    // (input is folded into b_bar already, matching this module's
    // discretization convention), y_t = c . h_t + d * x_t.
    for b in 0..batch_size {
        let mut h = vec![0.0f32; state_dim];
        for t in 0..seq_len {
            for s in 0..state_dim {
                h[s] = a_bars[[b, t, s]] * h[s] + b_bars[[b, t, s]];
            }
            let mut y = d_skip * x[[b, t]];
            for s in 0..state_dim {
                y += c[s] * h[s];
            }
            let got = output[[b, t]];
            assert!(
                (got - y).abs() < 1e-4,
                "batch {b} timestep {t}: parallel_ssm_batch={got}, manual reference={y}"
            );
        }
    }
}
