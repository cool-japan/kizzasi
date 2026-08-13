//! MQTT client for IoT/industrial sensor connectivity
//!
//! Provides enhanced MQTT client with:
//! - TLS/SSL support behind the opt-in `mqtt-tls` feature (see below)
//! - QoS levels (0, 1, 2)
//! - Retained messages
//! - Wildcard topic subscriptions
//! - Auto-reconnection with exponential backoff
//! - Message batching
//!
//! # TLS is opt-in
//!
//! Plain MQTT over TCP is pure Rust, and so is TLS: the `mqtt-tls` feature
//! selects rumqttc's `use-rustls-no-provider` transport, and this module
//! builds the `rustls::ClientConfig` explicitly with `oxitls-rustcrypto-provider`
//! (a pure-Rust RustCrypto `CryptoProvider`) injected directly -- never
//! rustls's default aws-lc-rs backend, so no C/assembly is compiled.
//! `rumqttc`'s `use-rustls-no-provider` feature does still hard-include
//! `rustls-native-certs`, which on Apple and Windows links the OS
//! certificate-store framework (Security.framework / schannel) -- that's a
//! link against an OS-provided library, not a C compilation, and kizzasi
//! itself never calls `load_native_certs`.
//!
//! TLS stays behind the non-default `mqtt-tls` feature regardless, because
//! turning it on is still a deliberate trust decision: the RustCrypto
//! provider is unaudited pure-Rust code, and its cipher suite list is
//! narrower than aws-lc-rs's -- 9 AEAD suites (ECDHE-{ECDSA,RSA} x
//! {AES-GCM,ChaCha20}, plus the three TLS 1.3 suites) with **no CBC suites**,
//! so a broker pinned to CBC-only cipher suites will fail the handshake.
//! None of that should happen implicitly. A configuration with
//! `use_tls = true` in a build without the `mqtt-tls` feature fails loudly
//! at [`MqttClient::connect`] -- it never downgrades to plaintext.

use crate::error::{IoError, IoResult};
use crate::stream::{SignalStream, StreamConfig};
use rumqttc::{AsyncClient, Broker, Event, EventLoop, Incoming, MqttOptions, QoS};
#[cfg(feature = "mqtt-tls")]
use rumqttc::{TlsConfiguration, Transport};
#[cfg(feature = "mqtt-tls")]
use rustls::{ClientConfig, RootCertStore};
#[cfg(feature = "mqtt-tls")]
use rustls_pki_types::pem::PemObject;
#[cfg(feature = "mqtt-tls")]
use rustls_pki_types::{CertificateDer, PrivateKeyDer};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
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
///
/// Always available so a configuration file round-trips in any build; it is
/// only *acted on* when the `mqtt-tls` feature is enabled (see the module
/// documentation).
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
    /// Shared with any `MqttStream` created via `stream()`. A plain
    /// `std::sync::Mutex` (not `tokio::sync::Mutex`) is deliberate: the
    /// critical sections that touch it are always short and never held
    /// across an `.await`, and using a std mutex lets the synchronous
    /// `SignalStream::read()` impl on `MqttStream` lock it directly instead
    /// of needing `blocking_lock()` (which panics if called from inside a
    /// Tokio runtime -- exactly where `read()` is likely to be called from).
    buffer: Arc<std::sync::Mutex<VecDeque<f32>>>,
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
            buffer: Arc::new(std::sync::Mutex::new(VecDeque::new())),
            message_buffer: Arc::new(Mutex::new(Vec::new())),
            active: Arc::new(Mutex::new(false)),
            client: None,
        }
    }

    /// Create a `MqttStream` sharing this client's live subscription
    /// buffer, so `SignalStream::read()` on the returned stream yields
    /// samples actually received over MQTT. `MqttStream::new` alone
    /// constructs a disconnected, standalone stream with its own private
    /// buffer (still useful for tests, or for manually feeding data via
    /// `push_data`); this is the constructor that wires it to a live
    /// client.
    pub fn stream(&self) -> MqttStream {
        MqttStream {
            config: self.stream_config.clone(),
            buffer: self.buffer.clone(),
            active: true,
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

        // Configure TLS if enabled. `use_tls = true` with no CA certificate
        // path must NEVER fall through to a plaintext connection -- that
        // would silently send credentials in the clear while the module
        // still claims "TLS/SSL support". Require an explicit CA cert path;
        // `MqttConfig::enable_tls(TlsConfig::default())` (the obvious way to
        // "turn on TLS" from the builder API) now fails loudly instead of
        // connecting over plain TCP.
        #[cfg(not(feature = "mqtt-tls"))]
        if self.config.use_tls {
            return Err(IoError::ConfigError(
                "TLS was requested (use_tls = true) but kizzasi-io was built without the \
                 `mqtt-tls` feature, so no TLS transport exists; refusing to fall back to a \
                 plaintext connection. Rebuild with `--features mqtt-tls` (pure Rust: rustls \
                 with the RustCrypto crypto provider, no C/assembly)."
                    .to_string(),
            ));
        }

        #[cfg(feature = "mqtt-tls")]
        if self.config.use_tls {
            let ca_path = self
                .config
                .tls_config
                .ca_cert_path
                .as_ref()
                .ok_or_else(|| {
                    IoError::ConfigError(
                    "TLS was requested (use_tls = true) but tls_config.ca_cert_path is not set; \
                     refusing to fall back to a plaintext connection. Set ca_cert_path to the \
                     broker's CA certificate."
                        .to_string(),
                )
                })?;

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

            // `TlsConfiguration::Simple` is deliberately never used here.
            // Its internal handling (rumqttc 0.30.1) builds the rustls
            // config via `rustls::ClientConfig::builder()`, which resolves
            // a process-global default `CryptoProvider` -- and panics
            // because it finds none: `use-rustls-no-provider` installs no
            // such default on purpose (later rumqttc versions return an
            // error instead of panicking, but the config is still
            // unusable). The `ClientConfig` is built explicitly instead,
            // with our own pure-Rust provider injected directly.
            let rustls_config =
                Self::build_rustls_client_config(&ca, client_auth.as_ref(), alpn.as_deref())?;

            options.set_transport(Transport::Tls(TlsConfiguration::Rustls(Arc::new(
                rustls_config,
            ))));
            info!("MQTT TLS/SSL enabled (pure-Rust rustls + RustCrypto provider)");
        }

        let (client, eventloop) = AsyncClient::new(options, 10);

        // Subscribe to all topics
        Self::subscribe_all(&client, &self.config.topics, self.config.qos).await?;

        // Mark as active
        *self.active.lock().await = true;

        self.client = Some(client.clone());

        let buffer = self.buffer.clone();
        let message_buffer = self.message_buffer.clone();
        let active = self.active.clone();
        let config = self.config.clone();

        // Spawn event loop handler with reconnection
        tokio::spawn(async move {
            Self::event_loop_task(eventloop, client, buffer, message_buffer, active, config).await;
        });

        Ok(())
    }

    /// Build a `rustls::ClientConfig` for the MQTT TLS transport, with the
    /// pure-Rust RustCrypto crypto provider (`oxitls-rustcrypto-provider`)
    /// injected directly instead of resolving a process-global default.
    ///
    /// Reproduces the trust semantics of rumqttc's `TlsConfiguration::Simple`
    /// (PEM-encoded CA certs into a `RootCertStore`; an optional PEM client
    /// cert/key pair for mTLS; optional ALPN protocols) without ever calling
    /// `rustls::ClientConfig::builder()` or `CryptoProvider::install_default`.
    #[cfg(feature = "mqtt-tls")]
    fn build_rustls_client_config(
        ca_pem: &[u8],
        client_auth: Option<&(Vec<u8>, Vec<u8>)>,
        alpn: Option<&[Vec<u8>]>,
    ) -> IoResult<ClientConfig> {
        let provider = Arc::new(oxitls_rustcrypto_provider::provider());

        let mut root_store = RootCertStore::empty();
        let ca_certs = CertificateDer::pem_slice_iter(ca_pem)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                IoError::ConfigError(format!("Failed to parse CA certificate PEM: {}", e))
            })?;
        root_store.add_parsable_certificates(ca_certs);
        if root_store.is_empty() {
            return Err(IoError::ConfigError(
                "No valid CA certificate found in tls_config.ca_cert_path".to_string(),
            ));
        }

        let builder = ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|e| {
                IoError::ConfigError(format!("Failed to select TLS protocol versions: {}", e))
            })?
            .with_root_certificates(root_store);

        let mut config = if let Some((cert_pem, key_pem)) = client_auth {
            let certs = CertificateDer::pem_slice_iter(cert_pem)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    IoError::ConfigError(format!("Failed to parse client certificate PEM: {}", e))
                })?;
            if certs.is_empty() {
                return Err(IoError::ConfigError(
                    "No valid client certificate found in tls_config.client_cert_path".to_string(),
                ));
            }
            let key = PrivateKeyDer::from_pem_slice(key_pem).map_err(|e| {
                IoError::ConfigError(format!("Failed to parse client key PEM: {}", e))
            })?;
            builder.with_client_auth_cert(certs, key).map_err(|e| {
                IoError::ConfigError(format!("Invalid client certificate/key: {}", e))
            })?
        } else {
            builder.with_no_client_auth()
        };

        if let Some(alpn) = alpn {
            config.alpn_protocols.extend_from_slice(alpn);
        }

        Ok(config)
    }

    /// Subscribe to every configured topic on the given client, logging and
    /// propagating any failure. Shared by the initial `connect()` and by
    /// `event_loop_task`'s re-subscription on every fresh `ConnAck`.
    async fn subscribe_all(client: &AsyncClient, topics: &[String], qos: QosLevel) -> IoResult<()> {
        let qos: QoS = qos.into();
        for topic in topics {
            client
                .subscribe(topic, qos)
                .await
                .map_err(|e| IoError::ConnectionFailed(format!("Subscribe failed: {}", e)))?;

            info!("MQTT subscribed to '{}' with QoS {:?}", topic, qos);
        }
        Ok(())
    }

    /// Whether a `ConnAck` means the broker discarded any prior
    /// subscription state and topics must be re-issued. Per the MQTT spec,
    /// `session_present` is only true when the broker resumed an existing
    /// (non-clean) session; with the default `clean_session = true` this is
    /// always false after a reconnect, which is exactly the case that used
    /// to leave the client silently subscribed to nothing.
    fn should_resubscribe(session_present: bool) -> bool {
        !session_present
    }

    /// Event loop task with auto-reconnection
    async fn event_loop_task(
        mut eventloop: EventLoop,
        client: AsyncClient,
        buffer: Arc<std::sync::Mutex<VecDeque<f32>>>,
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
                            let mut buf = buffer.lock().unwrap_or_else(|e| e.into_inner());
                            buf.extend(batch.drain(..));
                            last_batch_time = tokio::time::Instant::now();
                            debug!("MQTT batch flushed: {} samples", buf.len());
                        }
                    }

                    // Reset reconnect delay on successful message
                    reconnect_delay = Duration::from_millis(config.reconnect_delay_ms);
                }
                Ok(Event::Incoming(Incoming::ConnAck(connack))) => {
                    info!(
                        "MQTT connection acknowledged (session_present={})",
                        connack.session_present
                    );

                    // `connect()` only subscribes once, before this task is
                    // spawned. With the default `clean_session = true`, any
                    // reconnect after a transient drop makes the broker
                    // discard prior subscriptions, so the client silently
                    // stops receiving anything -- `is_connected()` still
                    // reports true and no error is ever surfaced. Re-issue
                    // every configured subscription whenever the broker
                    // reports a fresh session so a reconnect actually
                    // resumes the stream instead of going quietly dark.
                    if Self::should_resubscribe(connack.session_present) {
                        if let Err(e) =
                            Self::subscribe_all(&client, &config.topics, config.qos).await
                        {
                            error!("MQTT re-subscribe after reconnect failed: {}", e);
                        }
                    }
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
        let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        buffer.drain(..).collect()
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
///
/// A previous version had no way to receive real data: `MqttClient` never
/// held or fed a `MqttStream`, `push_data`'s only caller was its own
/// definition, and `read()` unconditionally returned a zero-filled array
/// presented as MQTT sensor data. Use `MqttClient::stream()` to get a
/// stream wired to a live subscription; `MqttStream::new` still exists for
/// standalone use (e.g. tests, or a custom async bridge that calls
/// `push_data` directly).
pub struct MqttStream {
    config: StreamConfig,
    buffer: Arc<std::sync::Mutex<VecDeque<f32>>>,
    active: bool,
}

impl MqttStream {
    /// Create a new, standalone MQTT stream with its own private buffer.
    /// Prefer `MqttClient::stream()` for a stream connected to a live MQTT
    /// subscription.
    pub fn new(config: StreamConfig) -> Self {
        Self {
            config,
            buffer: Arc::new(std::sync::Mutex::new(VecDeque::new())),
            active: true,
        }
    }

    /// Push data into the buffer (e.g. from a test, or a custom async
    /// bridge feeding this stream manually).
    pub fn push_data(&mut self, data: Vec<f32>) {
        let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        buffer.extend(data);
    }
}

impl SignalStream for MqttStream {
    fn read(&mut self) -> IoResult<Array1<f32>> {
        let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());

        if buffer.len() < self.config.buffer_size {
            // A previous version zero-padded a partial (or entirely empty)
            // buffer and returned Ok(Array1::zeros(..)), presenting
            // fabricated silence as real MQTT sensor data with no way for
            // the caller to distinguish it from genuine zero-valued
            // samples. Report the honest "not enough data yet" state
            // instead of a full buffer's worth of partially-fake data.
            return Err(IoError::BufferEmpty);
        }

        let mut result = Array1::zeros(self.config.buffer_size);
        for slot in result.iter_mut() {
            if let Some(value) = buffer.pop_front() {
                *slot = value;
            }
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

    // === Regression test: re-subscribe after reconnect (critical) ===
    //
    // The decision of *whether* to re-subscribe is a pure function of
    // `session_present` and is unit-testable directly. Whether the event
    // loop actually issues the SUBSCRIBE packets on a real reconnect
    // requires a live broker and is not exercised by this test suite; the
    // wiring (passing `client` into `event_loop_task` and calling
    // `subscribe_all` from the `ConnAck` arm) is verified by `cargo check`
    // and by code inspection.
    #[test]
    fn test_should_resubscribe_decision() {
        // clean_session=true (the default) always yields session_present
        // = false on every connect, which is exactly the case that used to
        // leave the client subscribed to nothing after a reconnect.
        assert!(MqttClient::should_resubscribe(false));
        // A broker that resumed an existing session already has our
        // subscriptions; no need to re-issue them.
        assert!(!MqttClient::should_resubscribe(true));
    }

    // === Regression test: TLS must never silently fall back to plaintext (critical) ===

    #[cfg(not(feature = "mqtt-tls"))]
    #[tokio::test]
    async fn test_connect_rejects_tls_when_built_without_tls_support() {
        // Same invariant as the `mqtt-tls` test below, for the default build:
        // asking for TLS in a build that has no TLS transport must fail, not
        // silently connect in the clear.
        let mqtt_config = MqttConfig::new("127.0.0.1", 1).enable_tls(TlsConfig::default());
        let mut client = MqttClient::new(mqtt_config, StreamConfig::new());

        match client.connect().await {
            Err(IoError::ConfigError(msg)) => {
                assert!(
                    msg.contains("mqtt-tls"),
                    "error should name the missing `mqtt-tls` feature, got: {msg}"
                );
            }
            other => panic!("expected IoError::ConfigError, got {other:?}"),
        }
    }

    #[cfg(feature = "mqtt-tls")]
    #[tokio::test]
    async fn test_connect_rejects_tls_without_ca_cert() {
        // `enable_tls(TlsConfig::default())` is the obvious way to "turn on
        // TLS" from the builder API, but leaves `ca_cert_path` unset. This
        // must fail loudly during `connect()` -- before any network I/O --
        // rather than silently connecting in plaintext.
        let mqtt_config = MqttConfig::new("127.0.0.1", 1).enable_tls(TlsConfig::default());
        let mut client = MqttClient::new(mqtt_config, StreamConfig::new());

        let result = client.connect().await;
        match result {
            Err(IoError::ConfigError(msg)) => {
                assert!(
                    msg.contains("ca_cert_path"),
                    "error should point at the missing ca_cert_path, got: {msg}"
                );
            }
            other => panic!("expected IoError::ConfigError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_connect_without_tls_does_not_require_ca_cert() {
        // Sanity check that the new validation is scoped to use_tls=true:
        // a plaintext config must not spuriously require a CA cert.
        // `connect()` only enqueues the subscribe requests and spawns the
        // event-loop task -- it does not itself wait for a live broker --
        // so this must succeed even though nothing listens on the target
        // port. Wrapped in a timeout as a safety net against any
        // unexpected blocking.
        let mqtt_config = MqttConfig::new("127.0.0.1", 1);
        assert!(!mqtt_config.use_tls);
        let mut client = MqttClient::new(mqtt_config, StreamConfig::new());

        let result = tokio::time::timeout(Duration::from_secs(5), client.connect())
            .await
            .expect("connect() should not hang waiting on a live broker");

        assert!(
            result.is_ok(),
            "plaintext connect() should not require a CA cert: {result:?}"
        );

        // Clean up the background reconnect task instead of leaving it
        // spinning against a nonexistent broker for the rest of the test
        // process's runtime.
        let _ = client.disconnect().await;
    }

    // === Regression tests: rustls `ClientConfig` is built explicitly with the
    // injected RustCrypto provider, not via `TlsConfiguration::Simple` (which
    // is unusable under `use-rustls-no-provider`; see `connect()`). These
    // exercise the actual PEM-decoding / `RootCertStore` / provider-builder
    // chain in `build_rustls_client_config`, not just that it type-checks. ===

    #[cfg(feature = "mqtt-tls")]
    const TEST_CA_CERT_PEM: &str = "-----BEGIN CERTIFICATE-----
MIIBiDCCAS+gAwIBAgIUeoHGPeFQO8lxcQhGTlJ3ZZg0/dQwCgYIKoZIzj0EAwIw
GjEYMBYGA1UEAwwPa2l6emFzaS10ZXN0LWNhMB4XDTI2MDgxMTA4NTE1MVoXDTM2
MDgwODA4NTE1MVowGjEYMBYGA1UEAwwPa2l6emFzaS10ZXN0LWNhMFkwEwYHKoZI
zj0CAQYIKoZIzj0DAQcDQgAEhw2xFaQ57pZs0lFoQ9sGbnDJc8mADbQv6G91yuX2
g3ZDUAIvlkHml9InR5pv6dnDfHZkZfodY/OOYt1fkBUYnaNTMFEwHQYDVR0OBBYE
FPS7CJg+jRo4ntVbJNxymklQSIXBMB8GA1UdIwQYMBaAFPS7CJg+jRo4ntVbJNxy
mklQSIXBMA8GA1UdEwEB/wQFMAMBAf8wCgYIKoZIzj0EAwIDRwAwRAIgJrfkq993
NpuQjuTYZeg+ClZWVJ+krriqptuvBo2jfMYCIE7ZZXdB/ATD/rRTKlPK3DdjC+6b
Leyy47RGnr27PA59
-----END CERTIFICATE-----
";

    #[cfg(feature = "mqtt-tls")]
    const TEST_CLIENT_CERT_PEM: &str = "-----BEGIN CERTIFICATE-----
MIIBfDCCASKgAwIBAgIUVXv95WtT6kneOoTUshGVIEoQQiYwCgYIKoZIzj0EAwIw
GjEYMBYGA1UEAwwPa2l6emFzaS10ZXN0LWNhMB4XDTI2MDgxMTA4NTE1MVoXDTM2
MDgwODA4NTE1MVowHjEcMBoGA1UEAwwTa2l6emFzaS10ZXN0LWNsaWVudDBZMBMG
ByqGSM49AgEGCCqGSM49AwEHA0IABKsJD1/O6XAn2szvNxusSzzP+6UtDe8MUUrb
OoHXsgt3eTU/Dlbdltx0NnwAeno6OHmVNe4YGdZEWlPjxUgMhb2jQjBAMB0GA1Ud
DgQWBBRMFMos6kifrbDsVKMz1VxAQxp0dzAfBgNVHSMEGDAWgBT0uwiYPo0aOJ7V
WyTccppJUEiFwTAKBggqhkjOPQQDAgNIADBFAiA+M7L2y8vWvStSQirL1UbWbXYM
ByPKyuLgycIeYlecsQIhAKgWljsV+JqIbzZ93s+x/WsdvPu1eDYe3FkaXCpirmlm
-----END CERTIFICATE-----
";

    #[cfg(feature = "mqtt-tls")]
    const TEST_CLIENT_KEY_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgypLIhqHpUBWe6uxW
9zFfYTiXHYl32X8yFL2yK+cxADShRANCAASrCQ9fzulwJ9rM7zcbrEs8z/ulLQ3v
DFFK2zqB17ILd3k1Pw5W3ZbcdDZ8AHp6Ojh5lTXuGBnWRFpT48VIDIW9
-----END PRIVATE KEY-----
";

    #[cfg(feature = "mqtt-tls")]
    #[test]
    fn test_build_rustls_client_config_parses_real_ca_pem() {
        // A real (P-256 ECDSA, self-signed) CA certificate must parse into a
        // non-empty `RootCertStore` and produce a usable `ClientConfig` --
        // proving the injected `oxitls_rustcrypto_provider::provider()`
        // actually satisfies `builder_with_provider(..).with_safe_default_protocol_versions()`,
        // not just that the call type-checks.
        let config =
            MqttClient::build_rustls_client_config(TEST_CA_CERT_PEM.as_bytes(), None, None)
                .expect("a valid self-signed CA PEM should build a ClientConfig");
        assert!(
            config.alpn_protocols.is_empty(),
            "no ALPN was requested, so none should be set"
        );
    }

    #[cfg(feature = "mqtt-tls")]
    #[test]
    fn test_build_rustls_client_config_with_client_auth_and_alpn() {
        // mTLS path: a client cert/key signed by the CA above, plus ALPN.
        let client_auth = (
            TEST_CLIENT_CERT_PEM.as_bytes().to_vec(),
            TEST_CLIENT_KEY_PEM.as_bytes().to_vec(),
        );
        let alpn = vec![b"mqtt".to_vec()];

        let config = MqttClient::build_rustls_client_config(
            TEST_CA_CERT_PEM.as_bytes(),
            Some(&client_auth),
            Some(&alpn),
        )
        .expect("a valid client cert/key PEM pair should build a ClientConfig with client auth");

        assert_eq!(
            config.alpn_protocols,
            vec![b"mqtt".to_vec()],
            "ALPN protocols should be carried through to the ClientConfig"
        );
    }

    #[cfg(feature = "mqtt-tls")]
    #[test]
    fn test_build_rustls_client_config_rejects_garbage_ca_pem() {
        // Not PEM at all: must be a clean `ConfigError`, not a panic.
        let result =
            MqttClient::build_rustls_client_config(b"this is not a PEM certificate", None, None);
        assert!(
            matches!(result, Err(IoError::ConfigError(_))),
            "garbage CA bytes should produce IoError::ConfigError, got {result:?}"
        );
    }

    // === Regression tests: MqttStream is no longer a disconnected shell (medium, id=309) ===

    #[test]
    fn test_mqtt_stream_read_errors_instead_of_fabricating_zeros() {
        let stream_config = StreamConfig::new().buffer_size(4);
        let mut stream = MqttStream::new(stream_config);

        // No data has been pushed: a previous version returned
        // Ok(Array1::zeros(..)) here, indistinguishable from a real
        // all-silence reading.
        let result = stream.read();
        assert!(
            matches!(result, Err(IoError::BufferEmpty)),
            "expected BufferEmpty, got {result:?}"
        );
    }

    #[test]
    fn test_mqtt_stream_read_requires_full_buffer_of_real_samples() {
        let stream_config = StreamConfig::new().buffer_size(4);
        let mut stream = MqttStream::new(stream_config);

        stream.push_data(vec![1.0, 2.0]); // fewer than buffer_size
        assert!(
            matches!(stream.read(), Err(IoError::BufferEmpty)),
            "a partial fill must not be silently zero-padded and returned as Ok"
        );

        stream.push_data(vec![3.0, 4.0]); // now exactly buffer_size
        let result = stream.read().expect("buffer now has buffer_size samples");
        assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[tokio::test]
    async fn test_mqtt_client_stream_shares_buffer_with_client() {
        let mqtt_config = MqttConfig::new("localhost", 1883);
        let stream_config = StreamConfig::new().buffer_size(4);
        let client = MqttClient::new(mqtt_config, stream_config);
        let mut stream = client.stream();

        // No data yet: honest error, not fabricated zeros.
        assert!(matches!(stream.read(), Err(IoError::BufferEmpty)));

        // Push through the stream handle -- this must land in the SAME
        // buffer the client itself owns, proving `stream()` is genuinely
        // wired up rather than constructing an independent copy.
        stream.push_data(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let result = stream.read().expect("buffer has >= buffer_size samples");
        assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

        // The leftover sample must be visible through the CLIENT's own
        // drain_buffer(), proving shared state rather than two
        // independent buffers.
        let remaining = client.drain_buffer().await;
        assert_eq!(remaining, vec![5.0]);
    }
}
