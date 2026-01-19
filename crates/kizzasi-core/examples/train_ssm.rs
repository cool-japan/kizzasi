//! Comprehensive SSM training example
//!
//! Demonstrates the complete training pipeline with:
//! - Time-series data loading
//! - Learning rate scheduling
//! - Metrics tracking
//! - Constraint-aware loss
//! - Validation and early stopping
//!
//! Run with: cargo run --example train_ssm

use kizzasi_core::{
    ConstraintLoss, CosineScheduler, DataLoaderConfig, KizzasiConfig, LRScheduler, Loss,
    SchedulerType, TimeSeriesDataLoader, TrainableSSM, Trainer, TrainingConfig,
};
use scirs2_core::ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    println!("=== Kizzasi SSM Training Example ===\n");

    // 1. Generate synthetic time-series data
    println!("Generating synthetic time-series data...");
    let (train_data, val_data) = generate_synthetic_data(1000, 3);
    println!(
        "Training data shape: {:?}, Validation data shape: {:?}\n",
        train_data.dim(),
        val_data.dim()
    );

    // 2. Configure data loaders
    println!("Configuring data loaders...");
    let dataloader_config = DataLoaderConfig {
        window_size: 32,
        horizon: 8,
        batch_size: 16,
        shuffle: true,
        overlap: 0.5,
        drop_last: false,
        num_workers: 1,
    };

    let train_loader = TimeSeriesDataLoader::new(train_data, dataloader_config.clone())?;
    let val_loader = TimeSeriesDataLoader::new(val_data, dataloader_config)?;

    println!(
        "Train batches: {}, Val batches: {}\n",
        train_loader.num_batches(),
        val_loader.num_batches()
    );

    // 3. Configure the model
    println!("Configuring SSM model...");
    let model_config = KizzasiConfig::new()
        .input_dim(3)
        .output_dim(3)
        .hidden_dim(128)
        .state_dim(16)
        .num_layers(4)
        .context_window(256);

    println!(
        "Model: {} layers, hidden_dim={}, state_dim={}\n",
        model_config.get_num_layers(),
        model_config.get_hidden_dim(),
        model_config.get_state_dim()
    );

    // 4. Configure training with scheduler
    println!("Configuring training...");
    let training_config = TrainingConfig {
        learning_rate: 1e-3,
        epochs: 20,
        batch_size: 16,
        grad_clip: Some(1.0),
        ..Default::default()
    }
    .with_scheduler(SchedulerType::Cosine {
        warmup_steps: 50,
        min_lr: 1e-6,
    })
    .with_validation_split(0.2)
    .with_early_stopping(5);

    println!(
        "Initial LR: {:.2e}, Epochs: {}",
        training_config.learning_rate, training_config.epochs
    );
    println!("Scheduler: Cosine with warmup\n");

    // 5. Create model and trainer
    println!("Initializing model and trainer...");
    let model = TrainableSSM::new(model_config, training_config.clone())?;
    let _trainer = Trainer::new(model, training_config)?;

    println!("Trainer initialized with metrics tracking and scheduler\n");

    // 6. Demonstrate learning rate schedule
    println!("Learning rate schedule preview (first 100 steps):");
    let scheduler = CosineScheduler::new(1e-3, 1000, 50).with_min_lr(1e-6);
    for step in (0..100).step_by(10) {
        let lr = scheduler.get_lr(step);
        println!("  Step {}: LR = {:.6e}", step, lr);
    }
    println!();

    // 7. Demonstrate constraint-aware loss
    println!("=== Constraint-Aware Training ===");
    println!("Constraint weight: 0.1");
    println!("Example: Penalizing predictions outside [-1, 1] range\n");

    let constraint_loss = ConstraintLoss::new(0.1);

    // Example constraint function
    let constraint_fn = |prediction: &candle_core::Tensor| -> Result<f32, kizzasi_core::CoreError> {
        // Compute how much the prediction violates the [-1, 1] range
        let pred_vec = prediction
            .flatten_all()
            .map_err(|e| kizzasi_core::CoreError::Generic(format!("Flatten failed: {}", e)))?
            .to_vec1::<f32>()
            .map_err(|e| kizzasi_core::CoreError::Generic(format!("To vec failed: {}", e)))?;

        let mut violation = 0.0;
        for &val in &pred_vec {
            if val < -1.0 {
                violation += (-1.0 - val).abs();
            } else if val > 1.0 {
                violation += (val - 1.0).abs();
            }
        }
        Ok(violation / pred_vec.len() as f32)
    };

    // Example: Create dummy prediction and compute constrained loss
    let device = candle_core::Device::Cpu;
    let dummy_pred = candle_core::Tensor::new(&[0.5f32, -0.8, 1.5], &device)?;
    let dummy_target = candle_core::Tensor::new(&[0.4f32, -0.7, 0.9], &device)?;

    let task_loss = Loss::mse(&dummy_pred, &dummy_target)?;
    let task_loss_val = task_loss.to_vec0::<f32>()?;

    let total_loss = constraint_loss.compute(&task_loss, &dummy_pred, constraint_fn)?;
    let total_loss_val = total_loss.to_vec0::<f32>()?;

    println!("Example prediction: [0.5, -0.8, 1.5]");
    println!("Task loss (MSE): {:.6}", task_loss_val);
    println!(
        "Constraint violation: {:.6}",
        (total_loss_val - task_loss_val) / 0.1
    );
    println!("Total loss: {:.6}\n", total_loss_val);

    // 8. Training metrics summary
    println!("=== Training Infrastructure Ready ===");
    println!("Components:");
    println!("  ✓ Time-series DataLoader with windowing and batching");
    println!("  ✓ Cosine learning rate scheduler with warmup");
    println!("  ✓ Automatic metrics tracking (loss, LR, gradients)");
    println!("  ✓ Validation with early stopping (patience=5)");
    println!("  ✓ Constraint-aware loss for constrained optimization");
    println!();

    // Note: Actual training loop would call trainer.fit()
    // For this example, we demonstrate the infrastructure setup
    println!("To run actual training:");
    println!("  trainer.fit(train_loader, Some(val_loader), Loss::mse)?;");
    println!();
    println!("Training infrastructure demonstration complete!");

    Ok(())
}

/// Generate synthetic time-series data
fn generate_synthetic_data(num_samples: usize, num_features: usize) -> (Array2<f32>, Array2<f32>) {
    use scirs2_core::convenience::uniform;
    use scirs2_core::ndarray::Array2;

    // Training data (80%)
    let train_samples = (num_samples as f32 * 0.8) as usize;
    let train_data = Array2::from_shape_fn((train_samples, num_features), |_| {
        (uniform() * 2.0 - 1.0) as f32
    });

    // Validation data (20%)
    let val_samples = num_samples - train_samples;
    let val_data = Array2::from_shape_fn((val_samples, num_features), |_| {
        (uniform() * 2.0 - 1.0) as f32
    });

    (train_data, val_data)
}
