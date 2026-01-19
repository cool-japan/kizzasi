//! Example: Model Versioning and Fallback
//!
//! This example demonstrates:
//! - Registering multiple model versions
//! - Health checking and monitoring
//! - Automatic fallback to stable versions
//! - Version management in production scenarios

use kizzasi_inference::versioning::FallbackStrategy;
use kizzasi_inference::{
    InferenceResult, ModelMetadata, ModelVersion, ModelVersionManager, VersioningConfig,
};
use std::time::Duration;

fn main() -> InferenceResult<()> {
    println!("=== Model Versioning and Fallback Example ===\n");

    // Step 1: Create version manager with configuration
    println!("1. Creating version manager with PreviousStable fallback strategy...");
    let config = VersioningConfig {
        fallback_strategy: FallbackStrategy::PreviousStable,
        health_check_interval: Duration::from_secs(60),
        max_error_rate: 0.1,
        max_latency_ms: 1000.0,
        auto_recovery: true,
    };
    let manager = ModelVersionManager::new(config);
    println!("   ✓ Manager created\n");

    // Step 2: Register multiple versions of a model
    println!("2. Registering model versions...");

    let v1_0_0 = ModelVersion::new(1, 0, 0);
    let metadata_v1 = ModelMetadata::new("agsp-predictor", v1_0_0.clone(), "Mamba2")
        .with_checksum("abc123")
        .add_metadata("training_date", "2024-01-15")
        .add_metadata("dataset", "production_v1");
    manager.register_version(metadata_v1, true)?; // Stable release
    println!("   ✓ Registered v1.0.0 (stable)");

    let v1_1_0 = ModelVersion::new(1, 1, 0);
    let metadata_v1_1 = ModelMetadata::new("agsp-predictor", v1_1_0.clone(), "Mamba2")
        .with_checksum("def456")
        .add_metadata("training_date", "2024-03-20")
        .add_metadata("dataset", "production_v2");
    manager.register_version(metadata_v1_1, true)?; // Stable release
    println!("   ✓ Registered v1.1.0 (stable)");

    let v1_2_0_beta = ModelVersion::new(1, 2, 0);
    let metadata_v1_2 = ModelMetadata::new("agsp-predictor", v1_2_0_beta.clone(), "Mamba2")
        .with_checksum("ghi789")
        .add_metadata("training_date", "2024-06-10")
        .add_metadata("dataset", "production_v3_experimental");
    manager.register_version(metadata_v1_2, false)?; // Beta release (not stable)
    println!("   ✓ Registered v1.2.0 (beta - not stable)\n");

    // Step 3: List all versions
    println!("3. Listing all registered versions:");
    let versions = manager.list_versions("agsp-predictor");
    for version in &versions {
        let metadata = manager.get_metadata("agsp-predictor", version).unwrap();
        let stable = if metadata.extra.contains_key("training_date") {
            "stable"
        } else {
            "unstable"
        };
        println!("   - v{} [{}]", version, stable);
    }
    println!();

    // Step 4: Set active version and simulate requests
    println!("4. Setting v1.2.0 (beta) as active version...");
    manager.set_active_version("agsp-predictor", v1_2_0_beta.clone())?;
    let active = manager.get_active_version("agsp-predictor").unwrap();
    println!("   ✓ Active version: v{}\n", active);

    // Step 5: Simulate successful requests
    println!("5. Simulating successful requests...");
    for i in 0..20 {
        let latency = 50.0 + (i as f64 * 5.0); // Gradually increasing latency
        manager.record_request("agsp-predictor", &v1_2_0_beta, latency, false)?;
    }
    let stats = manager.get_stats("agsp-predictor", &v1_2_0_beta).unwrap();
    println!("   Requests: {}", stats.total_requests);
    println!("   Avg Latency: {:.2}ms", stats.avg_latency_ms());
    println!("   Error Rate: {:.2}%\n", stats.error_rate() * 100.0);

    // Step 6: Health check
    println!("6. Performing health check on v1.2.0...");
    let health = manager.health_check("agsp-predictor", &v1_2_0_beta)?;
    println!("   Status: {:?}", health.status);
    println!("   Avg Latency: {:.2}ms", health.avg_latency_ms);
    println!("   Error Rate: {:.2}%", health.error_rate * 100.0);
    println!("   Usable: {}\n", health.is_usable());

    // Step 7: Simulate errors to trigger degradation
    println!("7. Simulating high error rate...");
    for _ in 0..30 {
        manager.record_request("agsp-predictor", &v1_2_0_beta, 150.0, true)?; // Errors
    }
    let health = manager.health_check("agsp-predictor", &v1_2_0_beta)?;
    println!("   Status after errors: {:?}", health.status);
    println!("   Error Rate: {:.2}%", health.error_rate * 100.0);
    println!("   Usable: {}\n", health.is_usable());

    // Step 8: Get fallback version
    println!("8. Getting fallback version (PreviousStable strategy)...");
    if let Some(fallback) = manager.get_fallback_version("agsp-predictor", &v1_2_0_beta) {
        println!("   ✓ Fallback version: v{}", fallback);
        let fallback_metadata = manager.get_metadata("agsp-predictor", &fallback).unwrap();
        println!("   Architecture: {}", fallback_metadata.architecture);
        println!(
            "   Checksum: {}",
            fallback_metadata.checksum.unwrap_or_default()
        );

        // Switch to fallback
        println!("\n9. Switching to fallback version...");
        manager.set_active_version("agsp-predictor", fallback.clone())?;
        let new_active = manager.get_active_version("agsp-predictor").unwrap();
        println!("   ✓ New active version: v{}", new_active);
    } else {
        println!("   ✗ No fallback available");
    }

    // Step 9: Demonstrate version compatibility
    println!("\n10. Checking version compatibility...");
    let v2_0_0 = ModelVersion::new(2, 0, 0);
    println!(
        "   v1.0.0 compatible with v1.1.0? {}",
        v1_0_0.is_compatible_with(&v1_1_0)
    );
    println!(
        "   v1.0.0 compatible with v2.0.0? {}",
        v1_0_0.is_compatible_with(&v2_0_0)
    );

    // Step 10: Version parsing from string
    println!("\n11. Parsing versions from strings...");
    let parsed = ModelVersion::parse("2.5.1")?;
    println!("   Parsed '2.5.1': v{}", parsed);
    println!(
        "   Major: {}, Minor: {}, Patch: {}",
        parsed.major, parsed.minor, parsed.patch
    );

    println!("\n=== Example completed successfully! ===");
    Ok(())
}
