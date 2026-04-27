//! MQTT client for IoT/industrial sensor connectivity
//!
//! Provides enhanced MQTT client with:
//! - TLS/SSL support
//! - QoS levels (0, 1, 2)
//! - Retained messages
//! - Wildcard topic subscriptions
//! - Auto-reconnection with exponential backoff
//! - Message batching

use crate::error::{IoError, IoResult};
use crate::stream::{SignalStream, StreamConfig};
use rumqttc::{
    AsyncClient, Broker, Event, EventLoop, Incoming, MqttOptions, QoS, TlsConfiguration, Transport,
};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

/// MQTT Quality of Service level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum QosLevel {
    /// At most once delivery (fire and forget)
    AtMostOnce = 0,
    /// At least once delivery (acknowledged)
    #[default]
    AtLeastOnce = 1,
    /// Exactly once delivery (assured)
    ExactlyOnce = 2,
}

impl From<QosLevel> for QoS {
    fn from(level: QosLevel) -> Self {
        match level {
            QosLevel::AtMostOnce => QoS::AtMostOnce,
            QosLevel::AtLeastOnce => QoS::AtLeastOnce,
            QosLevel::ExactlyOnce => QoS::ExactlyOnce,
        }
    }
}

/// TLS/SSL configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TlsConfig {
    /// Path to CA certificate file
    pub ca_cert_path: Option<String>,

    /// Path to client certificate file
    pub client_cert_path: Option<String>,

    /// Path to client key file
    pub client_key_path: Option<String>,

    /// ALPN protocols
    pub alpn: Option<Vec<String>>,
}

/// Configuration for MQTT connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqttConfig {
    /// MQTT broker host
    pub host: String,

    /// MQTT broker port
    pub port: u16,

    /// Client ID
    pub client_id: String,

    /// Topics to subscribe to (supports wildcards: +, #)
    pub topics: Vec<String>,

    /// QoS level
    #[serde(default)]
    pub qos: QosLevel,

    /// Keep alive interval in seconds
    #[serde(default = "default_keep_alive")]
    pub keep_alive_secs: u64,

    /// Enable TLS/SSL
    #[serde(default)]
    pub use_tls: bool,

    /// TLS configuration
    #[serde(default)]
    pub tls_config: TlsConfig,

    /// Username for authentication
    pub username: Option<String>,

    /// Password for authentication
    pub password: Option<String>,

    /// Enable retained message handling
    #[serde(default = "default_true")]
    pub handle_retained: bool,

    /// Enable auto-reconnection
    #[serde(default = "default_true")]
    pub auto_reconnect: bool,

    /// Reconnection delay (ms)
    #[serde(default = "default_reconnect_delay")]
    pub reconnect_delay_ms: u64,

    /// Maximum reconnection delay (ms)
    #[serde(default = "default_max_reconnect_delay")]
    pub max_reconnect_delay_ms: u64,

    /// Message batch size
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Batch timeout (ms)
    #[serde(default = "default_batch_timeout")]
    pub batch_timeout_ms: u64,

    /// Clean session flag
    #[serde(default = "default_true")]
    pub clean_session: bool,
}

fn default_keep_alive() -> u64 {
    30
}

fn default_true() -> bool {
    true
}

fn default_reconnect_delay() -> u64 {
    1000
}

fn default_max_reconnect_delay() -> u64 {
    30000
}

fn default_batch_size() -> usize {
    100
}

fn default_batch_timeout() -> u64 {
    100
}

impl Default for MqttConfig {
    fn default() -> Self {
        Self {
            host: "localhost".into(),
            port: 1883,
            client_id: "kizzasi-client".into(),
            topics: vec!["sensors/#".into()],
            qos: QosLevel::AtLeastOnce,
            keep_alive_secs: 30,
            use_tls: false,
            tls_config: TlsConfig::default(),
            username: None,
            password: None,
            handle_retained: true,
            auto_reconnect: true,
            reconnect_delay_ms: 1000,
            max_reconnect_delay_ms: 30000,
            batch_size: 100,
            batch_timeout_ms: 100,
            clean_session: true,
        }
    }
}

impl MqttConfig {
    /// Create a new MQTT configuration
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
            ..Default::default()
        }
    }

    /// Set topics (supports wildcards)
    pub fn topics(mut self, topics: Vec<String>) -> Self {
        self.topics = topics;
        self
    }

    /// Set a single topic
    pub fn topic(mut self, topic: &str) -> Self {
        self.topics = vec![topic.into()];
        self
    }

    /// Set the client ID
    pub fn client_id(mut self, id: &str) -> Self {
        self.client_id = id.into();
        self
    }

    /// Set QoS level
    pub fn qos(mut self, qos: QosLevel) -> Self {
        self.qos = qos;
        self
    }

    /// Enable TLS/SSL
    pub fn enable_tls(mut self, tls_config: TlsConfig) -> Self {
        self.use_tls = true;
        self.tls_config = tls_config;
        self
    }

    /// Set credentials
    pub fn credentials(mut self, username: String, password: String) -> Self {
        self.username = Some(username);
        self.password = Some(password);
        self
    }
}

/// MQTT message
#[derive(Debug, Clone)]
pub struct MqttMessage {
    /// Topic
    pub topic: String,

    /// Payload
    pub payload: Vec<u8>,

    /// QoS level
    pub qos: QosLevel,

    /// Retained flag
    pub retained: bool,
}

/// MQTT client for receiving sensor data
pub struct MqttClient {
    config: MqttConfig,
    stream_config: StreamConfig,
    buffer: Arc<Mutex<Vec<f32>>>,
    message_buffer: Arc<Mutex<Vec<MqttMessage>>>,
    active: Arc<Mutex<bool>>,
    client: Option<AsyncClient>,
}

impl MqttClient {
    /// Create a new MQTT client
    pub fn new(mqtt_config: MqttConfig, stream_config: StreamConfig) -> Self {
        Self {
            config: mqtt_config,
            stream_config,
            buffer: Arc::new(Mutex::new(Vec::new())),
            message_buffer: Arc::new(Mutex::new(Vec::new())),
            active: Arc::new(Mutex::new(false)),
            client: None,
        }
    }

    /// Connect to the MQTT broker and start receiving messages
    pub async fn connect(&mut self) -> IoResult<()> {
        let broker = Broker::tcp(self.config.host.as_str(), self.config.port);
        let mut options = MqttOptions::new(&self.config.client_id, broker);

        options.set_keep_alive(self.config.keep_alive_secs.try_into().unwrap_or(u16::MAX));
        options.set_clean_session(self.config.clean_session);

        // Set credentials if provided
        if let (Some(username), Some(password)) = (&self.config.username, &self.config.password) {
            options.set_credentials(username.clone(), password.clone());
        }

        // Configure TLS if enabled
        if self.config.use_tls {
            if let Some(ca_path) = &self.config.tls_config.ca_cert_path {
                let ca = std::fs::read(ca_path)
                    .map_err(|e| IoError::ConfigError(format!("Failed to read CA cert: {}", e)))?;

                let client_auth = if let (Some(cert_path), Some(key_path)) = (
                    &self.config.tls_config.client_cert_path,
                    &self.config.tls_config.client_key_path,
                ) {
                    let cert = std::fs::read(cert_path).map_err(|e| {
                        IoError::ConfigError(format!("Failed to read client cert: {}", e))
                    })?;
                    let key = std::fs::read(key_path).map_err(|e| {
                        IoError::ConfigError(format!("Failed to read client key: {}", e))
                    })?;
                    Some((cert, key))
                } else {
                    None
                };

                let alpn = self.config.tls_config.alpn.as_ref().map(|protocols| {
                    protocols
                        .iter()
                        .map(|s| s.as_bytes().to_vec())
                        .collect::<Vec<Vec<u8>>>()
                });

                let tls_config = TlsConfiguration::Simple {
                    ca,
                    alpn,
                    client_auth,
                };

                options.set_transport(Transport::Tls(tls_config));
                info!("MQTT TLS/SSL enabled");
            }
        }

        let (client, eventloop) = AsyncClient::new(options, 10);

        // Subscribe to all topics
        let qos: QoS = self.config.qos.into();
        for topic in &self.config.topics {
            client
                .subscribe(topic, qos)
                .await
                .map_err(|e| IoError::ConnectionFailed(format!("Subscribe failed: {}", e)))?;

            info!(
                "MQTT subscribed to '{}' with QoS {:?}",
                topic, self.config.qos
            );
        }

        // Mark as active
        *self.active.lock().await = true;

        self.client = Some(client.clone());

        let buffer = self.buffer.clone();
        let message_buffer = self.message_buffer.clone();
        let active = self.active.clone();
        let config = self.config.clone();

        // Spawn event loop handler with reconnection
        tokio::spawn(async move {
            Self::event_loop_task(eventloop, buffer, message_buffer, active, config).await;
        });

        Ok(())
    }

    /// Event loop task with auto-reconnection
    async fn event_loop_task(
        mut eventloop: EventLoop,
        buffer: Arc<Mutex<Vec<f32>>>,
        message_buffer: Arc<Mutex<Vec<MqttMessage>>>,
        active: Arc<Mutex<bool>>,
        config: MqttConfig,
    ) {
        let mut reconnect_delay = Duration::from_millis(config.reconnect_delay_ms);
        let max_delay = Duration::from_millis(config.max_reconnect_delay_ms);
        let batch_timeout = Duration::from_millis(config.batch_timeout_ms);
        let mut batch: Vec<f32> = Vec::with_capacity(config.batch_size);
        let mut last_batch_time = tokio::time::Instant::now();

        loop {
            if !*active.lock().await {
                break;
            }

            match eventloop.poll().await {
                Ok(Event::Incoming(Incoming::Publish(p))) => {
                    let topic_str = String::from_utf8_lossy(&p.topic).into_owned();
                    debug!(
                        "MQTT received on '{}': {} bytes",
                        topic_str,
                        p.payload.len()
                    );

                    // Handle retained messages
                    if p.retain && !config.handle_retained {
                        debug!("Skipping retained message");
                        continue;
                    }

                    // Store raw message
                    let msg = MqttMessage {
                        topic: topic_str.clone(),
                        payload: p.payload.to_vec(),
                        qos: match p.qos {
                            QoS::AtMostOnce => QosLevel::AtMostOnce,
                            QoS::AtLeastOnce => QosLevel::AtLeastOnce,
                            QoS::ExactlyOnce => QosLevel::ExactlyOnce,
                        },
                        retained: p.retain,
                    };

                    message_buffer.lock().await.push(msg);

                    // Try to parse payload as JSON array of floats
                    if let Ok(values) = serde_json::from_slice::<Vec<f32>>(&p.payload) {
                        batch.extend(values);

                        // Flush batch if full or timeout
                        if batch.len() >= config.batch_size
                            || last_batch_time.elapsed() >= batch_timeout
                        {
                            let mut buf = buffer.lock().await;
                            buf.extend(batch.drain(..));
                            last_batch_time = tokio::time::Instant::now();
                            debug!("MQTT batch flushed: {} samples", buf.len());
                        }
                    }

                    // Reset reconnect delay on successful message
                    reconnect_delay = Duration::from_millis(config.reconnect_delay_ms);
                }
                Ok(Event::Incoming(Incoming::ConnAck(_))) => {
                    info!("MQTT connection acknowledged");
                }
                Ok(Event::Incoming(Incoming::SubAck(_))) => {
                    debug!("MQTT subscription acknowledged");
                }
                Ok(Event::Incoming(Incoming::PingResp)) => {
                    debug!("MQTT ping response");
                }
                Ok(Event::Outgoing(_)) => {
                    // Outgoing events don't need handling
                }
                Err(e) => {
                    error!("MQTT connection error: {}", e);

                    if !config.auto_reconnect {
                        *active.lock().await = false;
                        break;
                    }

                    // Exponential backoff
                    warn!("Reconnecting in {:?}...", reconnect_delay);
                    tokio::time::sleep(reconnect_delay).await;
                    reconnect_delay = (reconnect_delay * 2).min(max_delay);
                }
                _ => {}
            }
        }

        info!("MQTT event loop terminated");
    }

    /// Get the current buffer contents
    pub async fn drain_buffer(&self) -> Vec<f32> {
        let mut buffer = self.buffer.lock().await;
        std::mem::take(&mut *buffer)
    }

    /// Get buffered messages
    pub async fn drain_messages(&self) -> Vec<MqttMessage> {
        let mut buffer = self.message_buffer.lock().await;
        std::mem::take(&mut *buffer)
    }

    /// Publish a message
    pub async fn publish(
        &self,
        topic: &str,
        payload: Vec<u8>,
        qos: QosLevel,
        retain: bool,
    ) -> IoResult<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| IoError::ConnectionFailed("Not connected".into()))?;

        client
            .publish(topic, qos.into(), retain, payload)
            .await
            .map_err(|e| IoError::SendFailed(format!("Publish failed: {}", e)))?;

        debug!("MQTT published to '{}' with QoS {:?}", topic, qos);
        Ok(())
    }

    /// Check if client is connected
    pub async fn is_connected(&self) -> bool {
        *self.active.lock().await
    }

    /// Disconnect from the broker
    pub async fn disconnect(&mut self) -> IoResult<()> {
        *self.active.lock().await = false;

        if let Some(client) = &self.client {
            client
                .disconnect()
                .await
                .map_err(|e| IoError::ConnectionFailed(format!("Disconnect failed: {}", e)))?;
        }

        self.client = None;
        info!("MQTT disconnected");
        Ok(())
    }

    /// Get the stream config
    pub fn stream_config(&self) -> &StreamConfig {
        &self.stream_config
    }

    /// Get the MQTT config
    pub fn mqtt_config(&self) -> &MqttConfig {
        &self.config
    }
}

/// Synchronous wrapper for MqttClient as SignalStream
pub struct MqttStream {
    config: StreamConfig,
    buffer: Vec<f32>,
    active: bool,
}

impl MqttStream {
    /// Create a new MQTT stream (placeholder for sync API)
    pub fn new(config: StreamConfig) -> Self {
        Self {
            config,
            buffer: Vec::new(),
            active: true,
        }
    }

    /// Push data into the buffer (called from async context)
    pub fn push_data(&mut self, data: Vec<f32>) {
        self.buffer.extend(data);
    }
}

impl SignalStream for MqttStream {
    fn read(&mut self) -> IoResult<Array1<f32>> {
        let size = self.config.buffer_size.min(self.buffer.len());
        if size == 0 {
            return Ok(Array1::zeros(self.config.buffer_size));
        }

        let data: Vec<f32> = self.buffer.drain(..size).collect();
        let mut result = Array1::zeros(self.config.buffer_size);
        for (i, val) in data.into_iter().enumerate() {
            result[i] = val;
        }
        Ok(result)
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn config(&self) -> &StreamConfig {
        &self.config
    }

    fn close(&mut self) -> IoResult<()> {
        self.active = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mqtt_config() {
        let config = MqttConfig::new("broker.example.com", 1883)
            .topic("sensors/temp")
            .client_id("test-client")
            .qos(QosLevel::ExactlyOnce);

        assert_eq!(config.host, "broker.example.com");
        assert_eq!(config.topics[0], "sensors/temp");
        assert_eq!(config.qos, QosLevel::ExactlyOnce);
    }

    #[test]
    fn test_qos_levels() {
        assert_eq!(QosLevel::AtMostOnce as u8, 0);
        assert_eq!(QosLevel::AtLeastOnce as u8, 1);
        assert_eq!(QosLevel::ExactlyOnce as u8, 2);
    }

    #[test]
    fn test_mqtt_config_credentials() {
        let config = MqttConfig::new("broker.example.com", 1883)
            .credentials("user".to_string(), "pass".to_string());

        assert_eq!(config.username, Some("user".to_string()));
        assert_eq!(config.password, Some("pass".to_string()));
    }
}
