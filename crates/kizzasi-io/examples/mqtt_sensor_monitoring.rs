//! MQTT Sensor Monitoring Example
//!
//! Demonstrates how to:
//! - Connect to an MQTT broker
//! - Subscribe to sensor topics
//! - Process incoming data streams
//! - Monitor data quality

use kizzasi_io::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MQTT Sensor Monitoring Example");
    println!("==============================\n");

    // Configure MQTT connection
    let mqtt_config = MqttConfig::new("localhost", 1883)
        .client_id("sensor_monitor")
        .topic("sensors/temperature")
        .qos(QosLevel::AtLeastOnce);

    println!("MQTT Configuration:");
    println!("  Broker: {}:{}", mqtt_config.host, mqtt_config.port);
    println!("  Client ID: {}", mqtt_config.client_id);
    println!("  Topics: {:?}", mqtt_config.topics);
    println!("  QoS: {:?}", mqtt_config.qos);
    println!("  Clean Session: {}", mqtt_config.clean_session);
    println!("  Keep Alive: {}s\n", mqtt_config.keep_alive_secs);

    // Example: Configure TLS for secure connection
    println!("TLS Configuration Example:");
    let tls_config = TlsConfig {
        ca_cert_path: Some("/path/to/ca.crt".to_string()),
        client_cert_path: None,
        client_key_path: None,
        alpn: None,
    };

    let _secure_config = mqtt_config.clone().enable_tls(tls_config);
    println!("  CA Certificate: /path/to/ca.crt");
    println!("  TLS enabled for secure communication\n");

    // Simulate sensor data processing
    println!("Simulating sensor data stream...\n");

    let mut health_monitor = HealthMonitor::new();
    let mut data_buffer: Vec<f32> = Vec::new();

    // Simulate receiving sensor data
    for i in 0..100 {
        // Simulate temperature reading (20-25°C with some noise)
        let noise = (i as f32 * 0.314159).cos() * 0.25; // Deterministic "noise"
        let temperature = 22.5 + (i as f32 * 0.1).sin() + noise;

        data_buffer.push(temperature);
        health_monitor.record_samples(&[temperature]);

        if data_buffer.len() >= 10 {
            // Process batch of readings
            let mean = data_buffer.iter().sum::<f32>() / data_buffer.len() as f32;
            let max = data_buffer.iter().fold(f32::MIN, |a, &b| a.max(b));
            let min = data_buffer.iter().fold(f32::MAX, |a, &b| a.min(b));

            println!(
                "Batch {}: Mean={:.2}°C, Min={:.2}°C, Max={:.2}°C",
                i / 10,
                mean,
                min,
                max
            );

            data_buffer.clear();
        }

        if i % 25 == 0 && i > 0 {
            // Check health status
            let health = health_monitor.health();
            let quality = health_monitor.signal_quality();

            println!("\nHealth Check:");
            println!("  Status: {:?}", health.status);
            println!("  Samples processed: {}", health.samples_processed);
            println!("  Signal SNR: {:.2} dB", quality.snr_db);
            println!("  Crest factor: {:.2}\n", quality.crest_factor);
        }
    }

    println!("\nExample: Wildcard topic subscription");
    let _wildcard_config = MqttConfig::new("localhost", 1883)
        .client_id("multi_sensor")
        .topic("sensors/#") // Subscribe to all sensor topics
        .qos(QosLevel::ExactlyOnce);

    println!("  Topic: sensors/#");
    println!("  This will receive messages from all sensor subtopics\n");

    println!("Example: Retained messages");
    let _retained_msg = MqttMessage {
        topic: "sensors/temperature/last".to_string(),
        payload: b"22.5".to_vec(),
        qos: QosLevel::AtLeastOnce,
        retained: true,
    };

    println!("  Publishing retained temperature reading");
    println!("  New subscribers will receive the last value immediately\n");

    println!("Example completed successfully!");
    println!("\nNote: This example uses simulated data.");
    println!("To use with a real MQTT broker:");
    println!("  1. Start an MQTT broker (e.g., mosquitto)");
    println!("  2. Update the broker URL");
    println!("  3. Uncomment the MqttStream creation code");

    Ok(())
}
