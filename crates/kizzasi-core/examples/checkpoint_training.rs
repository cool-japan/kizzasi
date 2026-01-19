//! Checkpoint Management Example
//!
//! Demonstrates saving and loading training checkpoints:
//! - Manual checkpoint saving
//! - Automatic epoch-based checkpoints
//! - Best model checkpoints
//! - Resuming training from checkpoints
//!
//! Run with: cargo run --example checkpoint_training

use kizzasi_core::*;
use scirs2_core::ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("=== Kizzasi Checkpoint Management Example ===\n");

    // 1. Create and train initial model
    println!("1. Training initial model for 5 epochs...");
    let (_initial_trainer, checkpoint_dir) = train_initial_model()?;
    println!("   Initial training complete!\n");

    // 2. Show checkpoint files
    println!("2. Checkpoint files created:");
    list_checkpoint_files(&checkpoint_dir)?;
    println!();

    // 3. Resume training from checkpoint
    println!("3. Resuming training from best checkpoint for 5 more epochs...");
    resume_and_continue_training(&checkpoint_dir)?;
    println!("   Resumed training complete!\n");

    // 4. Cleanup
    println!("4. Cleaning up temporary files...");
    std::fs::remove_dir_all(&checkpoint_dir)?;
    println!("   Done!\n");

    println!("=== Checkpoint Management Example Complete ===");

    Ok(())
}

/// Train initial model and save checkpoints
fn train_initial_model() -> Result<(Trainer, std::path::PathBuf), Box<dyn std::error::Error>> {
    // Generate synthetic training data (not used in this checkpoint-focused example)
    let _train_data = generate_synthetic_data(800, 3);

    // Configure model
    let model_config = KizzasiConfig::new()
        .input_dim(3)
        .output_dim(3)
        .hidden_dim(64)
        .state_dim(8)
        .num_layers(2);

    // Configure training
    let training_config = TrainingConfig {
        epochs: 5,
        learning_rate: 1e-3,
        batch_size: 16,
        ..Default::default()
    }
    .with_scheduler(SchedulerType::Cosine {
        warmup_steps: 10,
        min_lr: 1e-6,
    })
    .with_early_stopping(3);

    // Create model and trainer
    let model = TrainableSSM::new(model_config, training_config.clone())?;
    let mut trainer = Trainer::new(model, training_config)?;

    // Setup checkpoint directory
    let checkpoint_dir = std::env::temp_dir().join("kizzasi_checkpoints_example");
    std::fs::create_dir_all(&checkpoint_dir)?;

    // Simulate training loop with checkpoints
    for epoch in 0..5 {
        // Simulate some training metrics
        let train_loss = 1.0 - (epoch as f32 * 0.15);
        let val_loss = 0.9 - (epoch as f32 * 0.12);

        trainer.metrics_mut().record_train_loss(epoch, train_loss);
        trainer.metrics_mut().record_val_loss(epoch, val_loss);

        println!(
            "   Epoch {}: train_loss={:.4}, val_loss={:.4}",
            epoch, train_loss, val_loss
        );

        // Save checkpoint every 2 epochs
        if epoch % 2 == 0 {
            trainer.save_checkpoint_auto(&checkpoint_dir)?;
            println!("   → Saved epoch checkpoint");
        }

        // Save best checkpoint
        trainer.save_best_checkpoint(&checkpoint_dir)?;
    }

    // Save final checkpoint
    trainer.save_checkpoint(&checkpoint_dir, "final")?;
    println!("   → Saved final checkpoint");

    Ok((trainer, checkpoint_dir))
}

/// Resume training from checkpoint
fn resume_and_continue_training(
    checkpoint_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load the best checkpoint
    let model_config = KizzasiConfig::new()
        .input_dim(3)
        .output_dim(3)
        .hidden_dim(64)
        .state_dim(8)
        .num_layers(2);

    let mut trainer = Trainer::load_checkpoint(checkpoint_dir, "best", model_config)?;

    println!(
        "   Loaded checkpoint from step {} (best val loss: {:.4})",
        trainer.current_step(),
        trainer.metrics().best_val_loss().unwrap_or(0.0)
    );

    // Continue training for more epochs
    for epoch in 5..10 {
        let train_loss = 0.25 - ((epoch - 5) as f32 * 0.03);
        let val_loss = 0.30 - ((epoch - 5) as f32 * 0.04);

        trainer.metrics_mut().record_train_loss(epoch, train_loss);
        trainer.metrics_mut().record_val_loss(epoch, val_loss);

        println!(
            "   Epoch {}: train_loss={:.4}, val_loss={:.4}",
            epoch, train_loss, val_loss
        );

        trainer.save_best_checkpoint(checkpoint_dir)?;
    }

    // Save final resumed checkpoint
    trainer.save_checkpoint(checkpoint_dir, "resumed_final")?;
    println!("   → Saved resumed final checkpoint");

    Ok(())
}

/// List checkpoint files
fn list_checkpoint_files(
    checkpoint_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut entries: Vec<_> = std::fs::read_dir(checkpoint_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "safetensors" || ext == "json")
        })
        .collect();

    entries.sort_by_key(|e| e.path());

    for entry in entries {
        let path = entry.path();
        let size = entry.metadata()?.len();
        let filename = path.file_name().unwrap().to_string_lossy();

        println!("   - {} ({} bytes)", filename, size);
    }

    Ok(())
}

/// Generate synthetic time-series data
fn generate_synthetic_data(num_samples: usize, num_features: usize) -> Array2<f32> {
    use scirs2_core::convenience::uniform;

    Array2::from_shape_fn((num_samples, num_features), |_| {
        (uniform() * 2.0 - 1.0) as f32
    })
}
