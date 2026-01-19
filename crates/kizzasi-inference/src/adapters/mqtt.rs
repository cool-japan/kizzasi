//! MQTT adapter for publish/subscribe inference
//!
//! Provides MQTT-based inference for IoT and edge scenarios with reliable message delivery.

#![allow(clippy::arc_with_non_send_sync)]

use super::{InferenceMessage, InferenceResponse, NetworkAdapter};
use crate::error::{InferenceError, InferenceResult};
use crate::streaming::StreamingEngine;
use rumqttc::{AsyncClient, Event, EventLoop, MqttOptions, Packet, QoS};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info};

/// MQTT adapter for streaming inference
pub struct MqttAdapter {
    /// MQTT client
    client: AsyncClient,

    /// Event loop
    eventloop: Arc<RwLock<EventLoop>>,

    /// Streaming engine
    engine: Arc<RwLock<StreamingEngine>>,

    /// Input topic
    input_topic: String,

    /// Output topic
    output_topic: String,

    /// Running state
    running: Arc<RwLock<bool>>,
}

impl MqttAdapter {
    /// Create a new MQTT adapter
    ///
    /// # Arguments
    ///
    /// * `broker_url` - MQTT broker URL (e.g., "mqtt://broker.hivemq.com:1883")
    /// * `client_id` - Unique client identifier
    /// * `input_topic` - Topic to subscribe for inference requests
    /// * `output_topic` - Topic to publish inference responses
    /// * `engine` - Streaming inference engine
    pub fn new(
        broker_url: &str,
        client_id: &str,
        input_topic: impl Into<String>,
        output_topic: impl Into<String>,
        engine: StreamingEngine,
    ) -> InferenceResult<Self> {
        // Parse broker URL
        let url = broker_url
            .strip_prefix("mqtt://")
            .or_else(|| broker_url.strip_prefix("tcp://"))
            .unwrap_or(broker_url);

        let (host, port) = if let Some(idx) = url.find(':') {
            let (h, p) = url.split_at(idx);
            let port: u16 = p[1..]
                .parse()
                .map_err(|_| InferenceError::NetworkError("Invalid port".to_string()))?;
            (h.to_string(), port)
        } else {
            (url.to_string(), 1883)
        };

        let mut mqttoptions = MqttOptions::new(client_id, host, port);
        mqttoptions.set_keep_alive(Duration::from_secs(30));

        let (client, eventloop) = AsyncClient::new(mqttoptions, 10);

        Ok(Self {
            client,
            eventloop: Arc::new(RwLock::new(eventloop)),
            engine: Arc::new(RwLock::new(engine)),
            input_topic: input_topic.into(),
            output_topic: output_topic.into(),
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start processing MQTT messages
    pub async fn run(&self) -> InferenceResult<()> {
        // Subscribe to input topic
        self.client
            .subscribe(&self.input_topic, QoS::AtLeastOnce)
            .await
            .map_err(|e| InferenceError::NetworkError(e.to_string()))?;

        info!("MQTT adapter subscribed to topic: {}", self.input_topic);
        *self.running.write().await = true;

        // Process events
        while *self.running.read().await {
            let event = {
                let mut eventloop = self.eventloop.write().await;
                eventloop.poll().await
            };

            match event {
                Ok(Event::Incoming(Packet::Publish(publish))) => {
                    debug!(
                        "Received message on topic: {} ({} bytes)",
                        publish.topic,
                        publish.payload.len()
                    );

                    // Parse message
                    let request: InferenceMessage = match serde_json::from_slice(&publish.payload) {
                        Ok(req) => req,
                        Err(e) => {
                            error!("Failed to parse message: {}", e);
                            continue;
                        }
                    };

                    // Process inference
                    let start = std::time::Instant::now();
                    let input = request.to_array();

                    let output = {
                        let engine = self.engine.write().await;
                        match engine.step_async(input).await {
                            Ok(out) => out,
                            Err(e) => {
                                error!("Inference error: {}", e);
                                continue;
                            }
                        }
                    };

                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

                    // Create response
                    let response = InferenceResponse::new(
                        request.request_id,
                        output.to_vec(),
                        latency_ms,
                        output.len(),
                    );

                    // Publish response
                    let payload = match serde_json::to_vec(&response) {
                        Ok(p) => p,
                        Err(e) => {
                            error!("Failed to serialize response: {}", e);
                            continue;
                        }
                    };

                    if let Err(e) = self
                        .client
                        .publish(&self.output_topic, QoS::AtLeastOnce, false, payload)
                        .await
                    {
                        error!("Failed to publish response: {}", e);
                    }
                }
                Ok(Event::Incoming(_)) => {
                    // Ignore other packet types
                }
                Ok(Event::Outgoing(_)) => {
                    // Ignore outgoing events
                }
                Err(e) => {
                    error!("MQTT connection error: {}", e);
                    // Try to reconnect
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        }

        Ok(())
    }

    /// Publish a single message and get response
    pub async fn request(
        &self,
        input: Vec<f32>,
        request_id: &str,
    ) -> InferenceResult<InferenceResponse> {
        let msg = InferenceMessage::new(request_id, input);
        let payload = serde_json::to_vec(&msg)
            .map_err(|e| InferenceError::SerializationError(e.to_string()))?;

        self.client
            .publish(&self.input_topic, QoS::AtLeastOnce, false, payload)
            .await
            .map_err(|e| InferenceError::NetworkError(e.to_string()))?;

        // Note: In a real implementation, you would need to set up a response handler
        // This is a simplified version
        Err(InferenceError::NotImplemented(
            "Response handling not implemented in this example".to_string(),
        ))
    }
}

impl NetworkAdapter for MqttAdapter {
    async fn start(&mut self) -> InferenceResult<()> {
        self.run().await
    }

    async fn stop(&mut self) -> InferenceResult<()> {
        *self.running.write().await = false;
        info!("MQTT adapter stopped");
        Ok(())
    }

    fn is_running(&self) -> bool {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async { *self.running.read().await })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_mqtt_adapter_creation() {
        use crate::streaming::StreamConfig;

        let stream_config = StreamConfig::default();
        let engine = StreamingEngine::new(stream_config).unwrap();

        let adapter = MqttAdapter::new(
            "mqtt://test.mosquitto.org:1883",
            "test-client",
            "agsp/input",
            "agsp/output",
            engine,
        );

        assert!(adapter.is_ok());
        let adapter = adapter.unwrap();
        assert!(!adapter.is_running());
    }

    #[tokio::test]
    async fn test_mqtt_url_parsing() {
        use crate::streaming::StreamConfig;

        // Test mqtt:// format
        let stream_config1 = StreamConfig::default();
        let engine1 = StreamingEngine::new(stream_config1).unwrap();
        let adapter1 = MqttAdapter::new("mqtt://broker:1883", "client1", "in", "out", engine1);
        assert!(adapter1.is_ok());

        // Test tcp:// format
        let stream_config2 = StreamConfig::default();
        let engine2 = StreamingEngine::new(stream_config2).unwrap();
        let adapter2 = MqttAdapter::new("tcp://broker:1883", "client2", "in", "out", engine2);
        assert!(adapter2.is_ok());

        // Test no prefix format
        let stream_config3 = StreamConfig::default();
        let engine3 = StreamingEngine::new(stream_config3).unwrap();
        let adapter3 = MqttAdapter::new("broker:1883", "client3", "in", "out", engine3);
        assert!(adapter3.is_ok());
    }
}
