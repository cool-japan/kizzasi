//! Production Deployment Example
//!
//! This example demonstrates a complete production-ready deployment with:
//! - Model versioning and A/B testing
//! - Connection pooling for MQTT
//! - Telemetry and metrics collection
//! - Graceful degradation and error handling
//!
//! Run with: cargo run --example production_deployment --features full,async

use kizzasi::pool::{ConnectionFactory, ConnectionPool, PoolConfig};
use kizzasi::prelude::*;
use kizzasi::telemetry::{MetricEvent, MetricsCollector};
use kizzasi::versioning::{DeploymentStrategy, ModelMetadata, ModelRegistry, ModelVersion};
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "async")]
/// Simulated MQTT connection for demonstration
#[allow(dead_code)]
struct MqttConnection {
    id: usize,
    endpoint: String,
}

/// Factory for creating MQTT connections
struct MqttFactory {
    endpoint: String,
    next_id: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl ConnectionFactory<MqttConnection> for MqttFactory {
    async fn create(
        &self,
    ) -> std::result::Result<MqttConnection, Box<dyn std::error::Error + Send + Sync>> {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        Ok(MqttConnection {
            id,
            endpoint: self.endpoint.clone(),
        })
    }

    async fn validate(&self, _conn: &MqttConnection) -> bool {
        true
    }
}

/// Production inference service with all bells and whistles
struct ProductionService {
    registry: ModelRegistry,
    metrics: Arc<MetricsCollector>,
    #[cfg(feature = "async")]
    connection_pool: Option<Arc<ConnectionPool<MqttConnection>>>,
}

impl ProductionService {
    fn new() -> Self {
        Self {
            registry: ModelRegistry::new(),
            metrics: Arc::new(MetricsCollector::new("production_service")),
            #[cfg(feature = "async")]
            connection_pool: None,
        }
    }

    /// Initialize the service with models
    fn init(&mut self) -> Result<()> {
        println!("🚀 Initializing production service...\n");

        // Register model v1.0.0 (current production)
        let config_v1 = KizzasiConfig::new()
            .input_dim(64)
            .output_dim(64)
            .hidden_dim(256)
            .context_window(4096)
            .num_layers(4);

        let mut metadata_v1 = ModelMetadata {
            description: "Production baseline model".to_string(),
            author: Some("ML Team".to_string()),
            tags: vec!["production".to_string(), "baseline".to_string()],
            ..Default::default()
        };
        metadata_v1.metrics.insert("accuracy".to_string(), 0.92);

        let v1 = ModelVersion::with_metadata("1.0.0", config_v1, metadata_v1)?;
        self.registry.register(v1)?;

        println!("✓ Registered model v1.0.0 (production baseline)");

        // Register model v2.0.0 (improved model for canary deployment)
        let config_v2 = KizzasiConfig::new()
            .input_dim(64)
            .output_dim(64)
            .hidden_dim(512) // Doubled capacity
            .context_window(8192) // Doubled context
            .num_layers(6); // More layers

        let mut metadata_v2 = ModelMetadata {
            description: "Improved model with larger capacity".to_string(),
            author: Some("ML Team".to_string()),
            tags: vec!["canary".to_string(), "experimental".to_string()],
            ..Default::default()
        };
        metadata_v2.metrics.insert("accuracy".to_string(), 0.95);

        let v2 = ModelVersion::with_metadata("2.0.0", config_v2, metadata_v2)?;
        self.registry.register(v2)?;

        println!("✓ Registered model v2.0.0 (improved, canary)");

        // Deploy v2.0.0 with 10% canary traffic
        self.registry.deploy(
            "2.0.0",
            DeploymentStrategy::Canary {
                traffic_percent: 10,
            },
        )?;

        println!("✓ Deployed v2.0.0 with 10% canary traffic\n");

        Ok(())
    }

    #[cfg(feature = "async")]
    async fn init_connection_pool(&mut self) -> Result<()> {
        println!("🔌 Initializing connection pool...\n");

        let factory = Arc::new(MqttFactory {
            endpoint: "mqtt://broker.example.com:1883".to_string(),
            next_id: std::sync::atomic::AtomicUsize::new(0),
        });

        let config = PoolConfig::default()
            .with_min_connections(2)
            .with_max_connections(10)
            .with_acquire_timeout(tokio::time::Duration::from_secs(5));

        let pool = ConnectionPool::new(factory, config).await?;
        let stats = pool.stats().await;

        println!("✓ Connection pool initialized:");
        println!("  - Min connections: {}", stats.idle_connections);
        println!("  - Max connections: 10");
        println!("  - Created: {}\n", stats.total_created);

        self.connection_pool = Some(Arc::new(pool));

        Ok(())
    }

    /// Process a prediction request
    fn predict(&self, request_id: u64, input: &Array1<f32>) -> Result<Array1<f32>> {
        let start = Instant::now();

        // Select model based on A/B testing
        let model_version = self.registry.select_for_request(request_id)?;
        let mut predictor = model_version.create_predictor()?;

        // Perform prediction
        let result = predictor.step(input);

        // Record metrics
        let latency_us = start.elapsed().as_micros() as u64;
        self.metrics.record(MetricEvent::Prediction {
            latency_us,
            input_dim: input.len(),
            output_dim: predictor.output_dim(),
        });

        match result {
            Ok(output) => Ok(output),
            Err(e) => {
                self.metrics.record(MetricEvent::Error {
                    category: format!("{:?}", e.category()),
                });
                Err(e)
            }
        }
    }

    /// Display service statistics
    fn display_stats(&self) {
        println!("\n📊 Service Statistics:");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Model registry stats
        let registry_stats = self.registry.stats();
        println!("\n🔧 Model Registry:");
        println!("  - Total versions: {}", registry_stats.total_versions);
        println!(
            "  - Active version: {}",
            registry_stats
                .active_version
                .unwrap_or_else(|| "none".to_string())
        );
        println!(
            "  - Canary version: {}",
            registry_stats
                .canary_version
                .unwrap_or_else(|| "none".to_string())
        );

        // Metrics
        let metrics = self.metrics.snapshot();
        println!("\n📈 Performance Metrics:");
        println!("  - Total predictions: {}", metrics.total_predictions);
        println!("  - Total errors: {}", metrics.total_errors);
        println!("  - Average latency: {:.2} ms", metrics.avg_latency_ms);
        println!("  - P95 latency: {:.2} ms", metrics.p95_latency_ms);
        println!("  - P99 latency: {:.2} ms", metrics.p99_latency_ms);
        println!("  - Predictions/sec: {:.2}", metrics.predictions_per_second);
        println!("  - Error rate: {:.2}%", metrics.error_rate * 100.0);

        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    }

    /// Export metrics in Prometheus format
    fn export_prometheus(&self) {
        println!("\n📤 Prometheus Metrics Export:");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        let prometheus = self.metrics.export_prometheus();
        println!("{}", prometheus);
    }
}

fn main() -> Result<()> {
    println!("\n╔════════════════════════════════════════╗");
    println!("║   Production Deployment Example        ║");
    println!("╚════════════════════════════════════════╝\n");

    // Initialize service
    let mut service = ProductionService::new();
    service.init()?;

    #[cfg(feature = "async")]
    {
        // Run async initialization
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| KizzasiError::invalid_state(format!("Failed to create runtime: {}", e)))?;
        runtime.block_on(async { service.init_connection_pool().await })?;
    }

    // Simulate production traffic
    println!("🔄 Simulating production traffic...\n");

    let input = Array1::from_vec(vec![0.1; 64]);

    // Process 100 requests
    for request_id in 0..100 {
        match service.predict(request_id, &input) {
            Ok(_output) => {
                // Determine which model was used
                let model_version = service.registry.select_for_request(request_id)?;
                let version_str = model_version.version().to_string();

                if request_id % 20 == 0 {
                    println!(
                        "  Request #{}: ✓ Processed (model v{})",
                        request_id, version_str
                    );
                }
            }
            Err(e) => {
                println!("  Request #{}: ✗ Failed: {}", request_id, e);
            }
        }
    }

    println!("\n✓ Completed 100 requests\n");

    // Display statistics
    service.display_stats();

    // Export Prometheus metrics
    service.export_prometheus();

    // Demonstrate rollback scenario
    println!("\n⚠️  Simulating rollback scenario...");
    println!("  (Imagine canary version showed high error rate)\n");

    service.registry.rollback("1.0.0")?;
    println!("✓ Rolled back to v1.0.0\n");

    let registry_stats = service.registry.stats();
    println!(
        "  Active version: {}",
        registry_stats.active_version.unwrap()
    );
    println!(
        "  Canary version: {}",
        registry_stats
            .canary_version
            .unwrap_or_else(|| "none".to_string())
    );

    println!("\n✅ Production deployment example completed!\n");

    Ok(())
}
