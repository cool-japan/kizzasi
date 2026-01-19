//! Model checkpointing and persistence example
//!
//! This example demonstrates how to:
//! - Save predictor configuration to disk
//! - Load predictors from checkpoints
//! - Add metadata to checkpoints
//! - Version and manage models
//!
//! Run with:
//! ```bash
//! cargo run --example model_checkpointing
//! ```

use kizzasi::prelude::*;

fn main() -> Result<()> {
    println!("=== Kizzasi Model Checkpointing Example ===\n");

    // Create a temporary directory for our checkpoints
    let temp_dir = std::env::temp_dir();
    let checkpoint_dir = temp_dir.join("kizzasi_checkpoints");
    std::fs::create_dir_all(&checkpoint_dir)
        .map_err(|e| KizzasiError::Config(format!("Failed to create directory: {}", e)))?;

    println!("Checkpoint directory: {:?}\n", checkpoint_dir);

    // ========================================================================
    // 1. Basic checkpoint save/load
    // ========================================================================

    println!("1. Basic Checkpoint Operations");
    println!("{}", "=".repeat(45));

    // Create and configure a predictor
    let config = KizzasiConfig::new()
        .model_type(ModelType::Mamba2)
        .input_dim(5)
        .output_dim(5)
        .hidden_dim(128)
        .state_dim(16)
        .num_layers(3)
        .context_window(4096);

    let mut predictor = Kizzasi::new(config)?;
    println!("✓ Created predictor with custom configuration");

    // Run some predictions to establish state
    let input = array![0.1, 0.2, 0.3, 0.4, 0.5];
    for i in 0..10 {
        let output = predictor.step(&input)?;
        if i == 0 {
            println!("  First prediction: {:?}", output);
        }
    }
    println!("✓ Ran 10 prediction steps\n");

    // Save checkpoint
    let basic_checkpoint = checkpoint_dir.join("model_basic.checkpoint");
    predictor.save_checkpoint(&basic_checkpoint)?;
    println!("✓ Saved checkpoint to: {:?}", basic_checkpoint);

    // Load from checkpoint
    let restored = Kizzasi::load_checkpoint(&basic_checkpoint)?;
    println!("✓ Loaded predictor from checkpoint");
    println!("  - Input dim: {}", restored.config().get_input_dim());
    println!("  - Hidden dim: {}", restored.config().get_hidden_dim());
    println!("  - Num layers: {}\n", restored.config().get_num_layers());

    // ========================================================================
    // 2. Checkpoint with metadata
    // ========================================================================

    println!("2. Checkpoint with Metadata");
    println!("{}", "=".repeat(45));

    let mut audio_predictor = KizzasiBuilder::audio_preset().build()?;
    println!("✓ Created audio preset predictor");

    // Simulate training for 1000 steps
    let audio_input = array![0.5];
    for _ in 0..1000 {
        let _ = audio_predictor.step(&audio_input)?;
    }

    // Save with descriptive metadata
    let audio_checkpoint = checkpoint_dir.join("audio_model_v1.checkpoint");
    audio_predictor.save_checkpoint_with_metadata(
        &audio_checkpoint,
        "Audio model trained for 1000 steps",
        1000,
    )?;
    println!("✓ Saved checkpoint with metadata");

    // Load and inspect checkpoint metadata
    let checkpoint = PredictorCheckpoint::load_json(&audio_checkpoint)?;
    println!("\nCheckpoint Metadata:");
    println!("  - Version: {}", checkpoint.version);
    println!("  - Description: {:?}", checkpoint.metadata.description);
    println!("  - Step count: {}", checkpoint.metadata.step_count);
    println!("  - Created at: {}\n", checkpoint.metadata.created_at);

    // ========================================================================
    // 3. Multiple model versions
    // ========================================================================

    println!("3. Model Versioning");
    println!("{}", "=".repeat(45));

    // Create different model versions
    let versions = [
        (ModelType::Mamba, "v1", 64),
        (ModelType::Mamba2, "v2", 128),
        (ModelType::S4, "v3", 256),
    ];

    for (model_type, version, hidden_dim) in versions.iter() {
        let config = KizzasiConfig::new()
            .model_type(*model_type)
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(*hidden_dim)
            .num_layers(2);

        let predictor = Kizzasi::new(config)?;
        let checkpoint_path = checkpoint_dir.join(format!("model_{}.checkpoint", version));
        predictor.save_checkpoint(&checkpoint_path)?;

        println!(
            "✓ Saved {} model ({:?}, hidden_dim={})",
            version, model_type, hidden_dim
        );
    }
    println!();

    // Load and compare different versions
    println!("Comparing model versions:");
    for (_model_type, version, expected_hidden) in versions.iter() {
        let checkpoint_path = checkpoint_dir.join(format!("model_{}.checkpoint", version));
        let loaded = Kizzasi::load_checkpoint(&checkpoint_path)?;

        println!(
            "  {} - Model: {:?}, Hidden: {}, Match: {}",
            version,
            loaded.config().get_model_type(),
            loaded.config().get_hidden_dim(),
            loaded.config().get_hidden_dim() == *expected_hidden
        );
    }
    println!();

    // ========================================================================
    // 4. Preset configurations
    // ========================================================================

    println!("4. Saving Preset Configurations");
    println!("{}", "=".repeat(45));

    let presets = [
        ("robotics_6dof", KizzasiBuilder::robotics_preset(6).build()?),
        ("sensor_10ch", KizzasiBuilder::sensor_preset(10).build()?),
        (
            "lightweight",
            KizzasiBuilder::lightweight_preset(2, 2).build()?,
        ),
        ("video_256", KizzasiBuilder::video_preset(256).build()?),
    ];

    for (name, predictor) in presets.iter() {
        let checkpoint_path = checkpoint_dir.join(format!("preset_{}.checkpoint", name));
        predictor.save_checkpoint(&checkpoint_path)?;
        println!("✓ Saved '{}' preset configuration", name);
    }
    println!();

    // ========================================================================
    // 5. Custom metadata with JSON
    // ========================================================================

    println!("5. Advanced: Custom JSON Metadata");
    println!("{}", "=".repeat(45));

    let predictor = KizzasiBuilder::control_preset(8, 4).build()?;

    // Create custom metadata with experiment details
    let custom_metadata = serde_json::json!({
        "experiment": {
            "id": "exp_2026_001",
            "researcher": "Research Team",
            "purpose": "Control system evaluation"
        },
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        },
        "performance": {
            "loss": 0.0234,
            "accuracy": 0.956,
            "inference_time_ms": 12.3
        }
    });

    let checkpoint = PredictorCheckpoint::from_predictor(&predictor)
        .with_metadata("Control experiment", 5000)
        .with_custom_metadata(custom_metadata);

    let experiment_path = checkpoint_dir.join("experiment_checkpoint.json");
    checkpoint.save_json(&experiment_path)?;

    println!("✓ Saved checkpoint with custom JSON metadata");
    println!("\nCheckpoint contents preview:");
    let json_str = std::fs::read_to_string(&experiment_path)
        .map_err(|e| KizzasiError::Config(format!("Failed to read file: {}", e)))?;
    let lines: Vec<&str> = json_str.lines().take(20).collect();
    for line in lines {
        println!("  {}", line);
    }
    if json_str.lines().count() > 20 {
        println!("  ... (truncated)");
    }
    println!();

    // ========================================================================
    // 6. Checkpoint inspection without loading predictor
    // ========================================================================

    println!("6. Inspecting Checkpoints Without Loading Model");
    println!("{}", "=".repeat(45));

    println!("\nScanning checkpoint directory:");
    if let Ok(entries) = std::fs::read_dir(&checkpoint_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("checkpoint")
                || path.extension().and_then(|s| s.to_str()) == Some("json")
            {
                if let Ok(checkpoint) = PredictorCheckpoint::load_json(&path) {
                    println!("\n  File: {}", path.file_name().unwrap().to_string_lossy());
                    println!("    Model: {:?}", checkpoint.config.get_model_type());
                    println!(
                        "    Dims: {}x{}",
                        checkpoint.config.get_input_dim(),
                        checkpoint.config.get_output_dim()
                    );
                    println!("    Layers: {}", checkpoint.config.get_num_layers());
                    if let Some(ref desc) = checkpoint.metadata.description {
                        println!("    Description: {}", desc);
                    }
                }
            }
        }
    }
    println!();

    // ========================================================================
    // 7. Demonstrate limitations
    // ========================================================================

    println!("7. Understanding Checkpoint Limitations");
    println!("{}", "=".repeat(45));

    println!("\nCurrent limitations:");
    println!("  ⚠ SSM hidden state is NOT persisted");
    println!("    → Restored predictors start with fresh state");
    println!("  ⚠ Guardrails are NOT persisted");
    println!("    → Must manually re-add constraints after loading");
    println!("  ⚠ Model weights are NOT embedded in checkpoints");
    println!("    → Use 'weights_path' in config for external weights\n");

    // Demonstrate state reset
    let mut pred1 = KizzasiBuilder::lightweight_preset(2, 2).build()?;
    let input = array![0.1, 0.2];

    // Run predictions to build up state
    for _ in 0..5 {
        let _ = pred1.step(&input)?;
    }
    let output1 = pred1.step(&input)?;

    // Save and reload
    let temp_checkpoint = checkpoint_dir.join("temp_state_test.checkpoint");
    pred1.save_checkpoint(&temp_checkpoint)?;
    let mut pred2 = Kizzasi::load_checkpoint(&temp_checkpoint)?;

    // First prediction after load (fresh state)
    let output2 = pred2.step(&input)?;

    println!("State demonstration:");
    println!("  Original predictor (after 6 steps): {:?}", output1);
    println!("  Restored predictor (fresh state):   {:?}", output2);
    println!("  → Outputs differ because state was reset\n");

    // ========================================================================
    // Cleanup
    // ========================================================================

    println!("=== Example Complete ===");
    println!("\nCheckpoints saved to: {:?}", checkpoint_dir);
    println!("You can inspect these JSON files in any text editor.\n");

    // Optionally clean up (commented out to allow inspection)
    // std::fs::remove_dir_all(&checkpoint_dir)?;
    // println!("Cleaned up checkpoint directory");

    Ok(())
}
