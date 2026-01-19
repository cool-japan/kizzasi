//! Ensemble models with streaming inference
//!
//! This example demonstrates:
//! - Model ensembling with different architectures
//! - Different ensemble strategies
//! - Real-time prediction pipeline

use kizzasi_inference::{EnsembleBuilder, EnsembleStrategy, InferenceResult};
use kizzasi_model::mamba2::{Mamba2, Mamba2Config};
use kizzasi_model::rwkv::{Rwkv, RwkvConfig};
use kizzasi_model::s4::{S4Config, S4D};
use scirs2_core::ndarray::Array1;

fn main() -> InferenceResult<()> {
    println!("=== Ensemble Models Example ===\n");

    let test_input = Array1::from_vec(vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.1, 0.2,
        0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
    ]);

    // Test averaging ensemble
    println!("Testing strategy: Average");
    {
        let s4_model = Box::new(
            S4D::new(
                S4Config::new()
                    .input_dim(32)
                    .hidden_dim(128)
                    .state_dim(32)
                    .num_layers(3)
                    .diagonal(true),
            )
            .unwrap(),
        );

        let rwkv_model = Box::new(
            Rwkv::new(
                RwkvConfig::new()
                    .input_dim(32)
                    .hidden_dim(128)
                    .intermediate_dim(256)
                    .num_layers(3),
            )
            .unwrap(),
        );

        let mamba_model = Box::new(
            Mamba2::new(
                Mamba2Config::new()
                    .input_dim(32)
                    .hidden_dim(128)
                    .state_dim(32)
                    .num_layers(3),
            )
            .unwrap(),
        );

        let mut ensemble = EnsembleBuilder::new()
            .strategy(EnsembleStrategy::Average)
            .add_model(s4_model)
            .add_model(rwkv_model)
            .add_model(mamba_model)
            .build()?;

        let output = ensemble.step(&test_input)?;
        println!(
            "  Output range: [{:.3}, {:.3}]",
            output.iter().cloned().fold(f32::INFINITY, f32::min),
            output.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
        println!();
    }

    // Test weighted ensemble
    println!("Testing strategy: Weighted");
    {
        let s4_model = Box::new(
            S4D::new(
                S4Config::new()
                    .input_dim(32)
                    .hidden_dim(128)
                    .state_dim(32)
                    .num_layers(3)
                    .diagonal(true),
            )
            .unwrap(),
        );

        let rwkv_model = Box::new(
            Rwkv::new(
                RwkvConfig::new()
                    .input_dim(32)
                    .hidden_dim(128)
                    .intermediate_dim(256)
                    .num_layers(3),
            )
            .unwrap(),
        );

        let mamba_model = Box::new(
            Mamba2::new(
                Mamba2Config::new()
                    .input_dim(32)
                    .hidden_dim(128)
                    .state_dim(32)
                    .num_layers(3),
            )
            .unwrap(),
        );

        let mut ensemble = EnsembleBuilder::new()
            .strategy(EnsembleStrategy::Weighted)
            .add_model(s4_model)
            .add_model(rwkv_model)
            .add_model(mamba_model)
            .weights(vec![0.5, 0.3, 0.2])
            .build()?;

        let output = ensemble.step(&test_input)?;
        println!(
            "  Output range: [{:.3}, {:.3}]",
            output.iter().cloned().fold(f32::INFINITY, f32::min),
            output.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
        println!();
    }

    // Test voting ensemble
    println!("Testing strategy: Voting");
    {
        let s4_model = Box::new(
            S4D::new(
                S4Config::new()
                    .input_dim(32)
                    .hidden_dim(128)
                    .state_dim(32)
                    .num_layers(3)
                    .diagonal(true),
            )
            .unwrap(),
        );

        let rwkv_model = Box::new(
            Rwkv::new(
                RwkvConfig::new()
                    .input_dim(32)
                    .hidden_dim(128)
                    .intermediate_dim(256)
                    .num_layers(3),
            )
            .unwrap(),
        );

        let mamba_model = Box::new(
            Mamba2::new(
                Mamba2Config::new()
                    .input_dim(32)
                    .hidden_dim(128)
                    .state_dim(32)
                    .num_layers(3),
            )
            .unwrap(),
        );

        let mut ensemble = EnsembleBuilder::new()
            .strategy(EnsembleStrategy::Voting)
            .add_model(s4_model)
            .add_model(rwkv_model)
            .add_model(mamba_model)
            .build()?;

        let output = ensemble.step(&test_input)?;
        println!(
            "  Output range: [{:.3}, {:.3}]",
            output.iter().cloned().fold(f32::INFINITY, f32::min),
            output.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
        println!();
    }

    println!("=== Example Complete ===");
    Ok(())
}
