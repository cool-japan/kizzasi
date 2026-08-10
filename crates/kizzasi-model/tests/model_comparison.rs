//! Integration tests comparing different model architectures
//!
//! This test suite validates that all models:
//! - Can be created with valid configurations
//! - Produce consistent outputs
//! - Handle state management correctly
//! - Maintain numerically stable operations

use kizzasi_core::SignalPredictor;
use kizzasi_model::{
    mamba::{Mamba, MambaConfig},
    mamba2::{Mamba2, Mamba2Config},
    rwkv::{Rwkv, RwkvConfig},
    s4::{S4Config, S4D},
    transformer::{Transformer, TransformerConfig},
    AutoregressiveModel,
};
use scirs2_core::ndarray::Array1;

/// Test that all models can be created with small configurations
#[test]
fn test_all_models_creation() {
    // Mamba
    let mamba_config = MambaConfig::new().hidden_dim(64).state_dim(8).num_layers(2);
    let mamba = Mamba::new(mamba_config);
    assert!(mamba.is_ok());

    // Mamba2
    let mamba2_config = Mamba2Config::new()
        .hidden_dim(64)
        .num_heads(4)
        .num_layers(2);
    let mamba2 = Mamba2::new(mamba2_config);
    assert!(mamba2.is_ok());

    // RWKV
    let rwkv_config = RwkvConfig::new().hidden_dim(64).num_heads(4).num_layers(2);
    let rwkv = Rwkv::new(rwkv_config);
    assert!(rwkv.is_ok());

    // S4D
    let s4_config = S4Config::new().hidden_dim(64).state_dim(16).num_layers(2);
    let s4 = S4D::new(s4_config);
    assert!(s4.is_ok());

    // Transformer
    let transformer_config = TransformerConfig::new()
        .hidden_dim(64)
        .num_heads(4)
        .num_layers(2)
        .max_seq_len(128);
    let transformer = Transformer::new(transformer_config);
    assert!(transformer.is_ok());
}

/// Test that all models can perform forward pass
#[test]
fn test_all_models_forward() {
    let input = Array1::from_vec(vec![0.5]);

    // Mamba
    let mamba_config = MambaConfig::new().hidden_dim(32).state_dim(8).num_layers(1);
    let mut mamba = Mamba::new(mamba_config).unwrap();
    let output = mamba.step(&input);
    assert!(output.is_ok());
    assert_eq!(output.unwrap().len(), 1);

    // Mamba2
    let mamba2_config = Mamba2Config::new()
        .hidden_dim(32)
        .num_heads(4)
        .num_layers(1);
    let mut mamba2 = Mamba2::new(mamba2_config).unwrap();
    let output = mamba2.step(&input);
    assert!(output.is_ok());
    assert_eq!(output.unwrap().len(), 1);

    // RWKV
    let rwkv_config = RwkvConfig::new().hidden_dim(32).num_heads(4).num_layers(1);
    let mut rwkv = Rwkv::new(rwkv_config).unwrap();
    let output = rwkv.step(&input);
    assert!(output.is_ok());
    assert_eq!(output.unwrap().len(), 1);

    // S4D
    let s4_config = S4Config::new().hidden_dim(32).state_dim(8).num_layers(1);
    let mut s4 = S4D::new(s4_config).unwrap();
    let output = s4.step(&input);
    assert!(output.is_ok());
    assert_eq!(output.unwrap().len(), 1);

    // Transformer
    let transformer_config = TransformerConfig::new()
        .hidden_dim(32)
        .num_heads(4)
        .num_layers(1);
    let mut transformer = Transformer::new(transformer_config).unwrap();
    let output = transformer.step(&input);
    assert!(output.is_ok());
    assert_eq!(output.unwrap().len(), 1);
}

/// Test state persistence across multiple steps.
///
/// Strategy mirrors `comprehensive_tests::test_state_persistence`: after
/// running some steps, snapshot the state, advance, then restore the snapshot
/// twice and confirm subsequent `step()` calls produce identical output. This
/// validates determinism + `set_states` reproducibility without conflating
/// `get_states` round-trip fidelity (which has known fidelity limitations for
/// Mamba's auxiliary conv buffers — see issue #ssm-state-round-trip).
#[test]
fn test_state_persistence() {
    let mut mamba =
        Mamba::new(MambaConfig::new().hidden_dim(32).state_dim(8).num_layers(2)).unwrap();

    let inputs = vec![
        Array1::from_vec(vec![0.1]),
        Array1::from_vec(vec![0.2]),
        Array1::from_vec(vec![0.3]),
    ];

    // Build up some state.
    for input in &inputs {
        let _ = mamba.step(input).unwrap();
    }

    // Snapshot state.
    let snapshot = mamba.get_states();
    assert_eq!(snapshot.len(), 2); // 2 layers

    // Advance the model past the snapshot.
    for input in &inputs {
        let _ = mamba.step(input).unwrap();
    }

    // Confirm reset diverges from the snapshot trajectory.
    mamba.reset();
    let output_after_reset = mamba.step(&inputs[2]).unwrap();

    // Restore snapshot, step once.
    mamba.set_states(snapshot.clone()).unwrap();
    let output_first_restore = mamba.step(&inputs[2]).unwrap();

    // Reset trajectory must be different from the snapshot-restored trajectory.
    let diff_reset: f32 = output_first_restore
        .iter()
        .zip(output_after_reset.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff_reset > 1e-6,
        "Reset trajectory should differ from snapshot-restored trajectory (diff = {})",
        diff_reset
    );

    // Restore the same snapshot again — second restore must reproduce the
    // first restore exactly (determinism).
    mamba.set_states(snapshot).unwrap();
    let output_second_restore = mamba.step(&inputs[2]).unwrap();

    for i in 0..output_first_restore.len() {
        assert!(
            (output_first_restore[i] - output_second_restore[i]).abs() < 1e-4,
            "Mismatch at index {}: first={} vs second={}",
            i,
            output_first_restore[i],
            output_second_restore[i]
        );
    }
}

/// Test that `Mamba::get_states` / `set_states` round-trip is fully faithful.
///
/// This is the strict counterpart to `test_state_persistence`: it does not
/// merely check determinism on repeated restore — it checks that
/// `set_states(snapshot)` followed by `step(x)` produces the SAME output as
/// the original `step(x)` taken immediately after `get_states()`. A lossy
/// state snapshot (e.g. one that drops a frame of the conv history buffer)
/// will fail this test even though it can still pass the determinism check.
#[test]
fn test_state_roundtrip_fidelity() {
    let mut mamba =
        Mamba::new(MambaConfig::new().hidden_dim(32).state_dim(8).num_layers(2)).unwrap();
    let inputs: Vec<Array1<f32>> = (0..8)
        .map(|i| Array1::from_vec(vec![0.1 + i as f32 * 0.07]))
        .collect();

    // Run 5 steps to build up nontrivial state across both layers and the
    // causal conv history buffers, then snapshot.
    for input in inputs.iter().take(5) {
        let _ = mamba.step(input).unwrap();
    }
    let snapshot = mamba.get_states();
    let original_output = mamba.step(&inputs[5]).unwrap();

    // Advance the model so any state leakage from the snapshot is visible
    // immediately. Then restore the snapshot and replay the SAME input.
    let _ = mamba.step(&inputs[6]).unwrap();
    let _ = mamba.step(&inputs[7]).unwrap();
    mamba.set_states(snapshot).unwrap();
    let restored_output = mamba.step(&inputs[5]).unwrap();

    assert_eq!(original_output.len(), restored_output.len());
    for i in 0..original_output.len() {
        let diff = (original_output[i] - restored_output[i]).abs();
        assert!(
            diff < 1e-6,
            "State round-trip should be exact: idx {} original={} restored={} diff={}",
            i,
            original_output[i],
            restored_output[i],
            diff,
        );
    }
}

/// Test numerical stability with edge cases
#[test]
fn test_numerical_stability() {
    let config = MambaConfig::new().hidden_dim(32).state_dim(8).num_layers(2);
    let mut mamba = Mamba::new(config).unwrap();

    // Test with zero input
    let zero_input = Array1::from_vec(vec![0.0]);
    let output = mamba.step(&zero_input);
    assert!(output.is_ok());
    let output = output.unwrap();
    assert!(output.iter().all(|&x| x.is_finite()));

    // Test with large input
    let large_input = Array1::from_vec(vec![100.0]);
    let output = mamba.step(&large_input);
    assert!(output.is_ok());
    let output = output.unwrap();
    assert!(output.iter().all(|&x| x.is_finite()));

    // Test with small input
    let small_input = Array1::from_vec(vec![1e-6]);
    let output = mamba.step(&small_input);
    assert!(output.is_ok());
    let output = output.unwrap();
    assert!(output.iter().all(|&x| x.is_finite()));
}

/// Test that SSMs have infinite context window
#[test]
fn test_context_window() {
    // Use minimal configurations for faster tests
    let mamba = Mamba::new(MambaConfig::new().hidden_dim(32).state_dim(4).num_layers(1)).unwrap();
    let mamba2 = Mamba2::new(
        Mamba2Config::new()
            .hidden_dim(32)
            .num_heads(2)
            .num_layers(1),
    )
    .unwrap();
    let rwkv = Rwkv::new(RwkvConfig::new().hidden_dim(32).num_heads(2).num_layers(1)).unwrap();
    let s4 = S4D::new(S4Config::new().hidden_dim(32).state_dim(4).num_layers(1)).unwrap();

    // SSMs should have infinite context
    assert_eq!(mamba.context_window(), usize::MAX);
    assert_eq!(mamba2.context_window(), usize::MAX);
    assert_eq!(rwkv.context_window(), usize::MAX);
    assert_eq!(s4.context_window(), usize::MAX);

    // Transformer should have finite context
    let transformer = Transformer::new(
        TransformerConfig::new()
            .hidden_dim(32)
            .num_heads(2)
            .num_layers(1)
            .max_seq_len(512),
    )
    .unwrap();
    assert_eq!(transformer.context_window(), 512);
}

/// Test model type identification
#[test]
fn test_model_types() {
    use kizzasi_model::ModelType;

    // Use minimal configurations for faster tests
    let mamba = Mamba::new(
        MambaConfig::new()
            .hidden_dim(32)
            .state_dim(4)
            .num_layers(1)
            .mamba2(false),
    )
    .unwrap();
    assert_eq!(mamba.model_type(), ModelType::Mamba);

    let mamba2_via_flag = Mamba::new(
        MambaConfig::new()
            .hidden_dim(32)
            .state_dim(4)
            .num_layers(1)
            .mamba2(true),
    )
    .unwrap();
    assert_eq!(mamba2_via_flag.model_type(), ModelType::Mamba2);

    let mamba2 = Mamba2::new(
        Mamba2Config::new()
            .hidden_dim(32)
            .num_heads(2)
            .num_layers(1),
    )
    .unwrap();
    assert_eq!(mamba2.model_type(), ModelType::Mamba2);

    let rwkv = Rwkv::new(RwkvConfig::new().hidden_dim(32).num_heads(2).num_layers(1)).unwrap();
    assert_eq!(rwkv.model_type(), ModelType::Rwkv);

    let s4 = S4D::new(S4Config::new().hidden_dim(32).state_dim(4).num_layers(1)).unwrap();
    assert_eq!(s4.model_type(), ModelType::S4D);

    let transformer = Transformer::new(
        TransformerConfig::new()
            .hidden_dim(32)
            .num_heads(2)
            .num_layers(1),
    )
    .unwrap();
    assert_eq!(transformer.model_type(), ModelType::Transformer);
}

/// Test sequential processing maintains causality
#[test]
fn test_sequential_causality() {
    let mut model = Mamba::new(
        MambaConfig::new()
            .hidden_dim(64)
            .state_dim(16)
            .num_layers(3),
    )
    .unwrap();

    // Process sequence
    let sequence = vec![
        Array1::from_vec(vec![1.0]),
        Array1::from_vec(vec![2.0]),
        Array1::from_vec(vec![3.0]),
        Array1::from_vec(vec![4.0]),
    ];

    let mut outputs = Vec::new();
    for input in &sequence {
        outputs.push(model.step(input).unwrap());
    }

    // Process same sequence again from reset
    model.reset();
    for (i, input) in sequence.iter().enumerate() {
        let output = model.step(input).unwrap();
        // Should match previous outputs
        let diff: f32 = outputs[i]
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-5, "Output mismatch at position {}", i);
    }
}

/// Test configuration validation
#[test]
fn test_invalid_configurations() {
    // Invalid hidden_dim
    let config = MambaConfig {
        hidden_dim: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    // Invalid state_dim
    let config = MambaConfig {
        state_dim: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    // Invalid num_layers
    let config = MambaConfig {
        num_layers: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    // Invalid expand_factor
    let config = MambaConfig {
        expand_factor: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

/// Test that models handle multi-dimensional inputs correctly
#[test]
fn test_multidimensional_input() {
    let config = MambaConfig::new()
        .input_dim(3)
        .hidden_dim(64)
        .state_dim(16)
        .num_layers(2);

    let mut model = Mamba::new(config).unwrap();

    let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);
    let output = model.step(&input).unwrap();

    assert_eq!(output.len(), 3);
    assert!(output.iter().all(|&x| x.is_finite()));
}
