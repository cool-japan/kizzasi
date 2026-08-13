//! `SignalPredictor::step` must reject wrong-sized inputs instead of panicking.
//!
//! `ndarray`'s `Array1::dot(&Array2)` panics on a shape mismatch, and the input
//! length is entirely caller-controlled — `step` is the crate's primary public
//! entry point. Every model therefore validates the input length up front and
//! returns `CoreError::DimensionMismatch`.
//!
//! These tests would abort the whole test process (not merely fail) if any
//! model regressed to an unchecked `dot`.

use kizzasi_model::h3::{H3Config, H3};
use kizzasi_model::hybrid::{HybridConfig, HybridModel};
use kizzasi_model::mamba::{Mamba, MambaConfig};
use kizzasi_model::mamba2::{Mamba2, Mamba2Config};
use kizzasi_model::multimodal::{
    FusionStrategy, Modality, ModalityEncoderConfig, MultiModalConfig, MultiModalModel,
};
use kizzasi_model::neural_ode::{NeuralOdeConfig, NeuralOdeModel};
use kizzasi_model::rwkv::{Rwkv, RwkvConfig};
use kizzasi_model::rwkv5::{Rwkv5Config, Rwkv5Model};
use kizzasi_model::rwkv7::{Rwkv7Config, Rwkv7Model};
use kizzasi_model::s4::{S4Config, S4D};
use kizzasi_model::s5::{S5Config, S5};
use kizzasi_model::spiking::{SpikingConfig, SpikingNeuralNetwork};
use kizzasi_model::temporal_multiscale::{MultiScaleConfig, MultiScaleModel, ScaleFusion};
use kizzasi_model::transformer::{Transformer, TransformerConfig};
use kizzasi_model::{Array1, SignalPredictor};

const INPUT_DIM: usize = 8;

/// Assert that `step` accepts the configured width and rejects everything else.
fn assert_input_dim_enforced<M: SignalPredictor>(name: &str, model: &mut M, expected: usize) {
    model
        .step(&Array1::zeros(expected))
        .unwrap_or_else(|e| panic!("{name}: correctly-sized input must be accepted, got {e}"));

    for wrong in [0usize, 1, expected + 1, expected * 2 + 3] {
        if wrong == expected {
            continue;
        }
        let result = model.step(&Array1::zeros(wrong));
        assert!(
            result.is_err(),
            "{name}: step with a length-{wrong} input (expected {expected}) must return Err"
        );
    }
}

#[test]
fn mamba_rejects_wrong_input_length() {
    let mut model = Mamba::new(MambaConfig::tiny(INPUT_DIM)).expect("Mamba::new");
    assert_input_dim_enforced("Mamba", &mut model, INPUT_DIM);
}

#[test]
fn mamba2_rejects_wrong_input_length() {
    let config = Mamba2Config::new()
        .input_dim(INPUT_DIM)
        .hidden_dim(32)
        .num_heads(4)
        .num_layers(1);
    let mut model = Mamba2::new(config).expect("Mamba2::new");
    assert_input_dim_enforced("Mamba2", &mut model, INPUT_DIM);
}

#[test]
fn rwkv_rejects_wrong_input_length() {
    let base = RwkvConfig::new().hidden_dim(32).num_heads(4).num_layers(1);
    let config = RwkvConfig {
        input_dim: INPUT_DIM,
        intermediate_dim: 64,
        ..base
    };
    let mut model = Rwkv::new(config).expect("Rwkv::new");
    assert_input_dim_enforced("Rwkv", &mut model, INPUT_DIM);
}

#[test]
fn rwkv5_rejects_wrong_input_length() {
    let config = Rwkv5Config {
        input_dim: INPUT_DIM,
        hidden_dim: 32,
        num_heads: 4,
        head_dim: 8,
        intermediate_dim: 64,
        num_layers: 1,
        ..Default::default()
    };
    let mut model = Rwkv5Model::new(config).expect("Rwkv5Model::new");
    assert_input_dim_enforced("Rwkv5", &mut model, INPUT_DIM);
}

#[test]
fn rwkv7_rejects_wrong_input_length() {
    let config = Rwkv7Config {
        input_dim: INPUT_DIM,
        hidden_dim: 32,
        num_heads: 4,
        head_dim: 8,
        num_layers: 1,
        ..Default::default()
    };
    let mut model = Rwkv7Model::new(config).expect("Rwkv7Model::new");
    assert_input_dim_enforced("Rwkv7", &mut model, INPUT_DIM);
}

#[test]
fn s4d_rejects_wrong_input_length() {
    let config = S4Config::new()
        .input_dim(INPUT_DIM)
        .hidden_dim(32)
        .state_dim(8)
        .num_layers(1);
    let mut model = S4D::new(config).expect("S4D::new");
    assert_input_dim_enforced("S4D", &mut model, INPUT_DIM);
}

#[test]
fn s5_rejects_wrong_input_length() {
    let config = S5Config {
        state_dim: 8,
        ..S5Config::new(INPUT_DIM, 32, 1)
    };
    let mut model = S5::new(config).expect("S5::new");
    assert_input_dim_enforced("S5", &mut model, INPUT_DIM);
}

#[test]
fn h3_rejects_wrong_input_length() {
    let config = H3Config {
        ssm_dim: 8,
        ..H3Config::new(INPUT_DIM, 32, 1)
    };
    let mut model = H3::new(config).expect("H3::new");
    assert_input_dim_enforced("H3", &mut model, INPUT_DIM);
}

#[test]
fn transformer_rejects_wrong_input_length() {
    let config = TransformerConfig {
        input_dim: INPUT_DIM,
        hidden_dim: 32,
        num_heads: 4,
        head_dim: 8,
        ff_dim: 64,
        num_layers: 1,
        ..TransformerConfig::new()
    };
    let mut model = Transformer::new(config).expect("Transformer::new");
    assert_input_dim_enforced("Transformer", &mut model, INPUT_DIM);
}

#[test]
fn hybrid_rejects_wrong_input_length() {
    let config = HybridConfig::alternating(INPUT_DIM, 32, 2, 4);
    let mut model = HybridModel::new(config).expect("HybridModel::new");
    assert_input_dim_enforced("Hybrid", &mut model, INPUT_DIM);
}

#[test]
fn neural_ode_rejects_wrong_input_length() {
    let config = NeuralOdeConfig {
        input_dim: INPUT_DIM,
        hidden_dim: 16,
        num_layers: 1,
        integration_steps: 1,
        ..Default::default()
    };
    let mut model = NeuralOdeModel::new(config).expect("NeuralOdeModel::new");
    assert_input_dim_enforced("NeuralODE", &mut model, INPUT_DIM);
}

#[test]
fn spiking_network_rejects_wrong_input_length() {
    let config = SpikingConfig::new(INPUT_DIM, 16, INPUT_DIM, 1);
    let mut model = SpikingNeuralNetwork::new(config).expect("SpikingNeuralNetwork::new");
    assert_input_dim_enforced("SNN", &mut model, INPUT_DIM);
}

#[test]
fn multiscale_rejects_wrong_input_length() {
    let config = MultiScaleConfig {
        input_dim: INPUT_DIM,
        hidden_dim: 16,
        output_dim: INPUT_DIM,
        num_scales: 2,
        scale_factors: vec![1, 4],
        fusion: ScaleFusion::Concatenate,
        context_length: 64,
    };
    let mut model = MultiScaleModel::new(config).expect("MultiScaleModel::new");
    assert_input_dim_enforced("MultiScale", &mut model, INPUT_DIM);
}

#[test]
fn multimodal_rejects_wrong_input_length() {
    let config = MultiModalConfig {
        fusion_dim: 16,
        fusion_strategy: FusionStrategy::Addition,
        output_dim: INPUT_DIM,
        modalities: vec![
            ModalityEncoderConfig {
                modality: Modality::Sensor,
                input_dim: 4,
                projection_dim: 16,
                num_layers: 1,
            },
            ModalityEncoderConfig {
                modality: Modality::Control,
                input_dim: 4,
                projection_dim: 16,
                num_layers: 1,
            },
        ],
        context_length: 64,
    };
    let mut model = MultiModalModel::new(config).expect("MultiModalModel::new");
    assert_input_dim_enforced("MultiModal", &mut model, INPUT_DIM);
}
