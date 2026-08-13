//! MQTT adapter for publish/subscribe inference
//!
//! Provides MQTT-based inference for IoT and edge scenarios with reliable message delivery.

#![allow(clippy::arc_with_non_send_sync)]

use super::{InferenceMessage, InferenceResponse, NetworkAdapter};
use crate::error::{InferenceError, InferenceResult};
use crate::streaming::StreamingEngine;
use rumqttc::{AsyncClient, Broker, Event, EventLoop, MqttOptions, Packet, QoS};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{oneshot, watch, Mutex, RwLock};
use tracing::{debug, error, info, warn};

/// Default request timeout when waiting for an MQTT response.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

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

    /// Running state, and the signal `stop()` uses to interrupt an in-flight
    /// `run()` call blocked on `eventloop.poll()` (or its reconnect-backoff
    /// sleep). See `GrpcAdapter`/`WebSocketAdapter` for the same pattern.
    running: watch::Sender<bool>,

    /// Pending request/response correlations.
    ///
    /// Maps a `request_id` to the oneshot channel that the `run()` loop will
    /// complete when a matching response arrives on the output topic.
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<InferenceResponse>>>>,
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

        let broker = Broker::tcp(host, port);
        let mut mqttoptions = MqttOptions::new(client_id, broker);
        mqttoptions.set_keep_alive(30u16);

        let (client, eventloop) = AsyncClient::new(mqttoptions, 10);
        let (running, _) = watch::channel(false);

        Ok(Self {
            client,
            eventloop: Arc::new(RwLock::new(eventloop)),
            engine: Arc::new(RwLock::new(engine)),
            input_topic: input_topic.into(),
            output_topic: output_topic.into(),
            running,
            pending: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Start processing MQTT messages, until [`NetworkAdapter::stop`] is
    /// called.
    pub async fn run(&self) -> InferenceResult<()> {
        // Subscribe to input topic
        self.client
            .subscribe(&self.input_topic, QoS::AtLeastOnce)
            .await
            .map_err(|e| InferenceError::NetworkError(e.to_string()))?;

        info!("MQTT adapter subscribed to topic: {}", self.input_topic);
        let _ = self.running.send_replace(true);
        let mut shutdown_rx = self.running.subscribe();

        // Process events
        loop {
            let event = tokio::select! {
                biased;

                _ = shutdown_rx.wait_for(|running| !*running) => {
                    break;
                }
                event = async {
                    let mut eventloop = self.eventloop.write().await;
                    eventloop.poll().await
                } => {
                    event
                }
            };

            match event {
                Ok(Event::Incoming(Packet::Publish(publish))) => {
                    let topic = String::from_utf8_lossy(&publish.topic).into_owned();
                    debug!(
                        "Received message on topic: {} ({} bytes)",
                        topic,
                        publish.payload.len()
                    );

                    // Parse message as an inference request.
                    let request: InferenceMessage = match serde_json::from_slice(&publish.payload) {
                        Ok(req) => req,
                        Err(e) => {
                            error!("Failed to parse message on {}: {}", topic, e);
                            continue;
                        }
                    };

                    let request_id = request.request_id.clone();

                    // Process inference
                    let start = std::time::Instant::now();
                    let input = request.to_array();

                    let outputs = {
                        let engine = self.engine.write().await;
                        match engine
                            .step_async_with(input, &(&request.config).into())
                            .await
                        {
                            Ok(out) => out,
                            Err(e) => {
                                error!("Inference error for request {}: {}", request_id, e);
                                continue;
                            }
                        }
                    };

                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

                    // Create response
                    let (output, num_tokens) = super::flatten_outputs(&outputs);
                    let response =
                        InferenceResponse::new(request_id.clone(), output, latency_ms, num_tokens);

                    // If there is a pending in-process caller waiting for this request_id,
                    // resolve it directly via the oneshot channel — no MQTT round-trip needed.
                    let pending_sender = {
                        let mut pending = self.pending.lock().await;
                        pending.remove(&request_id)
                    };
                    if let Some(tx) = pending_sender {
                        if tx.send(response.clone()).is_err() {
                            warn!(
                                "Pending request {} was dropped before response could be delivered",
                                request_id
                            );
                        }
                    }

                    // Also publish to the output topic so external MQTT subscribers
                    // (if any) receive the response as well.
                    let payload = match serde_json::to_vec(&response) {
                        Ok(p) => p,
                        Err(e) => {
                            error!("Failed to serialize response for {}: {}", request_id, e);
                            continue;
                        }
                    };

                    if let Err(e) = self
                        .client
                        .publish(&self.output_topic, QoS::AtLeastOnce, false, payload)
                        .await
                    {
                        error!("Failed to publish response for {}: {}", request_id, e);
                    }
                }
                Ok(Event::Incoming(_)) => {
                    // Ignore other packet types (ConnAck, SubAck, PingResp, etc.)
                }
                Ok(Event::Outgoing(_)) => {
                    // Ignore outgoing events
                }
                Err(e) => {
                    error!("MQTT connection error: {}", e);
                    // Try to reconnect, but stay interruptible: a `stop()`
                    // during the backoff must not have to wait out the full
                    // 5 seconds before `run()` returns.
                    tokio::select! {
                        biased;
                        _ = shutdown_rx.wait_for(|running| !*running) => break,
                        _ = tokio::time::sleep(Duration::from_secs(5)) => {}
                    }
                }
            }
        }

        let _ = self.running.send_replace(false);
        info!("MQTT adapter stopped");
        Ok(())
    }

    /// Publish a single inference request and wait for the response.
    ///
    /// This method publishes to the `input_topic` so the [`Self::run`] loop can
    /// process it, then awaits the response on an internal oneshot channel.
    /// The call times out after the internal `REQUEST_TIMEOUT` (30 seconds).
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::NetworkError`] if the publish fails,
    /// or [`InferenceError::Timeout`] if no response arrives within the
    /// deadline.
    pub async fn request(
        &self,
        input: Vec<f32>,
        request_id: &str,
    ) -> InferenceResult<InferenceResponse> {
        // Register a oneshot channel *before* publishing so there is no window
        // in which a very fast run() loop could respond before we register.
        let (tx, rx) = oneshot::channel::<InferenceResponse>();
        {
            let mut pending = self.pending.lock().await;
            pending.insert(request_id.to_string(), tx);
        }

        // Build and publish the inference request.
        let msg = InferenceMessage::new(request_id, input);
        let payload = serde_json::to_vec(&msg)
            .map_err(|e| InferenceError::SerializationError(e.to_string()))?;

        if let Err(e) = self
            .client
            .publish(&self.input_topic, QoS::AtLeastOnce, false, payload)
            .await
        {
            // Clean up the pending entry if the publish fails.
            let mut pending = self.pending.lock().await;
            pending.remove(request_id);
            return Err(InferenceError::NetworkError(e.to_string()));
        }

        // Wait for run() to dispatch the response via the oneshot channel.
        match tokio::time::timeout(REQUEST_TIMEOUT, rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                // The sender was dropped (run() exited or encountered an error).
                Err(InferenceError::NetworkError(
                    "Response channel closed before response was delivered".to_string(),
                ))
            }
            Err(_timeout) => {
                // Timed out — remove the stale pending entry.
                let mut pending = self.pending.lock().await;
                pending.remove(request_id);
                Err(InferenceError::Timeout(format!(
                    "No response for request {} within {}s",
                    request_id,
                    REQUEST_TIMEOUT.as_secs()
                )))
            }
        }
    }
}

impl NetworkAdapter for MqttAdapter {
    async fn start(&self) -> InferenceResult<()> {
        self.run().await
    }

    async fn stop(&self) -> InferenceResult<()> {
        let _ = self.running.send_replace(false);
        info!("MQTT adapter stop requested");
        Ok(())
    }

    fn is_running(&self) -> bool {
        *self.running.borrow()
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

    /// Regression: `NetworkAdapter::stop()` used to flip a flag `run()`
    /// never read, and even after fixing the read, the loop's own 5-second
    /// reconnect backoff was not itself interruptible. `start()` must now
    /// actually resolve promptly after `stop()`, whether it is currently
    /// blocked polling the (unreachable) broker or backing off from a
    /// connection error.
    ///
    /// Points at a just-closed local port (rather than a real broker
    /// hostname) so the connection is refused immediately and
    /// deterministically, without depending on external network access.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_network_adapter_stop_unblocks_start() {
        use crate::streaming::StreamConfig;
        use std::net::TcpListener as StdTcpListener;

        let closed_addr = {
            let listener =
                StdTcpListener::bind("127.0.0.1:0").expect("OS should assign a free port");
            listener.local_addr().expect("local_addr must be set")
            // `listener` is dropped here, closing the port.
        };

        let engine = StreamingEngine::new(StreamConfig::default()).unwrap();
        let adapter = Arc::new(
            MqttAdapter::new(
                &format!("mqtt://{}", closed_addr),
                "stop-test-client",
                "agsp/input",
                "agsp/output",
                engine,
            )
            .expect("adapter construction must succeed"),
        );

        let adapter_for_task = adapter.clone();
        let handle = tokio::spawn(async move { adapter_for_task.start().await });

        // Poll rather than a single fixed sleep: robust against scheduling
        // jitter when many tests run concurrently.
        let mut became_running = false;
        for _ in 0..100 {
            if adapter.is_running() {
                became_running = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(
            became_running,
            "adapter should report running after start()"
        );

        adapter.stop().await.expect("stop must succeed");

        let result = tokio::time::timeout(std::time::Duration::from_secs(3), handle).await;
        assert!(
            result.is_ok(),
            "start() must resolve promptly after stop(), well under the 5s reconnect backoff"
        );
        assert!(!adapter.is_running());
    }
}
