//! Custom Model Example
//!
//! Demonstrates advanced Kizzasi features including:
//! - Custom model configurations
//! - Hot-swapping between different model types
//! - Lazy initialization for resource efficiency
//! - Configuration file loading (TOML/YAML)
//! - Model comparison and benchmarking

use kizzasi::{
    Kizzasi, KizzasiBuilder, KizzasiConfig, KizzasiResult as Result, LazyKizzasi, ModelType,
};
use scirs2_core::ndarray::{array, Array1};

#[cfg(feature = "config-files")]
use std::env;

fn main() -> Result<()> {
    println!("=== Kizzasi Custom Model Example ===\n");

    // Example 1: Custom architecture from scratch
    println!("1. Building custom architecture...");
    custom_architecture_example()?;

    // Example 2: Hot-swapping between models
    println!("\n2. Hot-swapping between models...");
    hot_swap_example()?;

    // Example 3: Lazy initialization
    println!("\n3. Lazy initialization demo...");
    lazy_initialization_example()?;

    // Example 4: Configuration file loading
    #[cfg(feature = "config-files")]
    {
        println!("\n4. Configuration file loading...");
        config_file_example()?;
    }

    // Example 5: Model comparison
    println!("\n5. Comparing different model types...");
    model_comparison_example()?;

    println!("\n=== All examples completed successfully! ===");
    Ok(())
}

/// Example 1: Building a custom architecture from scratch
fn custom_architecture_example() -> Result<()> {
    // Create a custom high-capacity model for complex signal processing
    let mut custom_config = KizzasiBuilder::custom_preset()
        .input_dim(16) // 16-channel input
        .output_dim(16)
        .hidden_dim(512) // Large hidden dimension
        .state_dim(64) // Large state for long-term dependencies
        .num_layers(8) // Deep architecture
        .context_window(16384) // Very large context
        .model_type(ModelType::Mamba2) // Use Mamba2 for efficiency
        .build()?;

    // Test the custom model
    let input: Array1<f32> = Array1::from_vec((0..16).map(|i| i as f32 * 0.1).collect());
    let output = custom_config.step(&input)?;

    println!("  Custom model created:");
    println!("    Input dim: {}", custom_config.input_dim());
    println!("    Hidden dim: {}", custom_config.hidden_dim());
    println!("    Layers: {}", custom_config.num_layers());
    println!("    Context: {}", custom_config.context_window());
    println!("  Output shape: {:?}", output.shape());

    Ok(())
}

/// Example 2: Hot-swapping between different model types at runtime
fn hot_swap_example() -> Result<()> {
    let mut predictor = KizzasiBuilder::new()
        .model_type(ModelType::Mamba2)
        .input_dim(4)
        .output_dim(4)
        .hidden_dim(128)
        .state_dim(16)
        .num_layers(3)
        .build()?;

    println!("  Initial model: {:?}", predictor.model_type());

    // Make some predictions with Mamba2
    let input = array![0.1, 0.2, 0.3, 0.4];
    let output1 = predictor.step(&input)?;
    println!("  Mamba2 output: {:?}", output1);

    // Hot-swap to RWKV while keeping dimensions
    let rwkv_config = KizzasiConfig::new()
        .model_type(ModelType::Rwkv)
        .input_dim(4)
        .output_dim(4)
        .hidden_dim(256) // Can change internal architecture
        .state_dim(32)
        .num_layers(4);

    predictor.hot_swap(rwkv_config, false)?;
    println!("  Swapped to: {:?}", predictor.model_type());

    // Make prediction with new model
    let output2 = predictor.step(&input)?;
    println!("  RWKV output: {:?}", output2);

    // Swap to S4
    let s4_config = KizzasiConfig::new()
        .model_type(ModelType::S4)
        .input_dim(4)
        .output_dim(4)
        .hidden_dim(128)
        .state_dim(16)
        .num_layers(2);

    predictor.hot_swap(s4_config, false)?;
    println!("  Swapped to: {:?}", predictor.model_type());

    let output3 = predictor.step(&input)?;
    println!("  S4 output: {:?}", output3);

    Ok(())
}

/// Example 3: Lazy initialization for resource efficiency
fn lazy_initialization_example() -> Result<()> {
    // Create lazy predictor - no resources allocated yet
    let lazy_config = KizzasiConfig::new()
        .model_type(ModelType::Mamba2)
        .input_dim(8)
        .output_dim(8)
        .hidden_dim(256)
        .state_dim(16)
        .num_layers(4);

    let lazy_predictor = LazyKizzasi::new(lazy_config);

    println!("  Lazy predictor created");
    println!("  Initialized? {}", lazy_predictor.is_initialized());

    // Can access config without initialization
    println!(
        "  Config accessible: input_dim = {}",
        lazy_predictor.config().get_input_dim()
    );

    // First prediction triggers initialization
    let input: Array1<f32> = Array1::from_vec((0..8).map(|i| i as f32 * 0.05).collect());
    println!("  Making first prediction (triggers initialization)...");
    let output = lazy_predictor.step(&input)?;

    println!("  Initialized? {}", lazy_predictor.is_initialized());
    println!("  Output: {:?}", output);

    // Subsequent predictions are fast (already initialized)
    let output2 = lazy_predictor.step(&input)?;
    println!("  Second prediction (already initialized): {:?}", output2);

    Ok(())
}

/// Example 4: Configuration file loading
#[cfg(feature = "config-files")]
fn config_file_example() -> Result<()> {
    use kizzasi::{ConfigFile, ConfigLoader};

    // Create a config and save it to TOML
    let config = KizzasiConfig::new()
        .model_type(ModelType::Mamba2)
        .input_dim(5)
        .output_dim(5)
        .hidden_dim(256)
        .state_dim(16)
        .num_layers(4)
        .context_window(4096);

    let toml_path = env::temp_dir().join("kizzasi_custom.toml");
    config.to_file(&toml_path)?;
    println!("  Saved config to: {:?}", toml_path);

    // Create config file manually and save as YAML
    let custom_file = ConfigFile {
        model_type: Some("Mamba2".into()),
        input_dim: Some(10),
        output_dim: Some(10),
        hidden_dim: Some(512),
        state_dim: Some(32),
        num_layers: Some(6),
        context_window: Some(8192),
        weights_path: None,
        expansion_factor: None,
        head_dim: None,
        num_heads: None,
    };

    let yaml_path = env::temp_dir().join("kizzasi_custom.yaml");
    custom_file.save(&yaml_path)?;
    println!("  Saved custom config to: {:?}", yaml_path);

    // Load from TOML
    let loaded_toml = KizzasiConfig::from_file(&toml_path)?;
    println!(
        "  Loaded from TOML: input_dim={}, hidden_dim={}",
        loaded_toml.get_input_dim(),
        loaded_toml.get_hidden_dim()
    );

    // Load from YAML
    let loaded_yaml = KizzasiConfig::from_file(&yaml_path)?;
    println!(
        "  Loaded from YAML: input_dim={}, hidden_dim={}",
        loaded_yaml.get_input_dim(),
        loaded_yaml.get_hidden_dim()
    );

    // Create predictor from loaded config
    let mut predictor = Kizzasi::new(loaded_yaml)?;
    let input: Array1<f32> = Array1::from_vec((0..10).map(|i| i as f32 * 0.1).collect());
    let output = predictor.step(&input)?;
    println!(
        "  Predictor from YAML config works! Output shape: {:?}",
        output.shape()
    );

    // Cleanup
    let _ = std::fs::remove_file(toml_path);
    let _ = std::fs::remove_file(yaml_path);

    Ok(())
}

/// Example 5: Comparing different model types
fn model_comparison_example() -> Result<()> {
    let input = array![0.1, 0.2, 0.3];
    let models = vec![
        ("Mamba", ModelType::Mamba),
        ("Mamba2", ModelType::Mamba2),
        ("S4", ModelType::S4),
        ("RWKV", ModelType::Rwkv),
    ];

    println!("  Comparing models with same input:");
    println!("  Input: {:?}", input);

    for (name, model_type) in models {
        let mut predictor = KizzasiBuilder::new()
            .model_type(model_type)
            .input_dim(3)
            .output_dim(3)
            .hidden_dim(64)
            .state_dim(8)
            .num_layers(2)
            .build()?;

        let output = predictor.step(&input)?;
        println!("  {} output: {:?}", name, output);

        // Multi-step prediction
        let multi_output = predictor.predict_n(&input, 3)?;
        println!("    3-step predictions shape: {:?}", multi_output.shape());
    }

    Ok(())
}

/// Example 6: Advanced custom configuration with presets
#[allow(dead_code)]
fn preset_customization_example() -> Result<()> {
    // Start with a preset and customize it
    let custom_audio = KizzasiBuilder::audio_preset()
        .hidden_dim(512) // Increase capacity
        .num_layers(6) // Deeper network
        .context_window(16384) // Larger context
        .build()?;

    println!("  Customized audio preset:");
    println!("    Hidden dim: {}", custom_audio.hidden_dim());
    println!("    Layers: {}", custom_audio.num_layers());

    // Custom robotics with different architecture
    let custom_robotics = KizzasiBuilder::robotics_preset(6)
        .model_type(ModelType::Rwkv) // Try different model type
        .hidden_dim(256) // Increase from default
        .build()?;

    println!("  Customized robotics preset:");
    println!("    Model: {:?}", custom_robotics.model_type());
    println!("    Hidden dim: {}", custom_robotics.hidden_dim());

    Ok(())
}
