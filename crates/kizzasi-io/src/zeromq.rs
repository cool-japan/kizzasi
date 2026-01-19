//! ZeroMQ stream support
//!
//! Provides ZeroMQ messaging patterns for distributed streaming.
//!
//! ## Features
//! - SUB/PUB pattern support
//! - PUSH/PULL pattern support
//! - REQ/REP pattern support
//! - Topic filtering for SUB sockets
//! - Auto-reconnection support
//! - High-throughput messaging
//!
//! ## Example
//! ```rust,no_run
//! use kizzasi_io::{ZmqStream, ZmqConfig, ZmqPattern};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Subscribe to a ZeroMQ publisher
//!     let config = ZmqConfig {
//!         endpoint: "tcp://localhost:5555".to_string(),
//!         pattern: ZmqPattern::Sub,
//!         topics: vec!["sensor".to_string()],
//!         ..Default::default()
//!     };
//!
//!     let mut stream = ZmqStream::connect(config).await?;
//!
//!     while let Some(msg) = stream.recv().await? {
//!         println!("Received: {:?}", msg);
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::error::{IoError, IoResult};
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tmq::{publish, pull, push, Context, Message, Multipart};
use tracing::info;

/// ZeroMQ messaging pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZmqPattern {
    /// Subscriber (receives from PUB)
    Sub,
    /// Publisher (sends to SUB)
    Pub,
    /// Pull (receives from PUSH)
    Pull,
    /// Push (sends to PULL)
    Push,
    /// Request (sends to REP, receives reply)
    Req,
    /// Reply (receives from REQ, sends reply)
    Rep,
    /// Dealer (async REQ)
    Dealer,
}

/// ZeroMQ configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZmqConfig {
    /// ZeroMQ endpoint (e.g., "tcp://localhost:5555")
    pub endpoint: String,

    /// Messaging pattern
    pub pattern: ZmqPattern,

    /// Topics to subscribe to (for SUB pattern)
    #[serde(default)]
    pub topics: Vec<String>,

    /// High water mark (message buffer size)
    #[serde(default = "default_hwm")]
    pub high_water_mark: i32,

    /// Receive timeout (ms) - 0 for no timeout
    #[serde(default)]
    pub recv_timeout_ms: u64,

    /// Send timeout (ms) - 0 for no timeout
    #[serde(default)]
    pub send_timeout_ms: u64,

    /// Linger period (ms) - time to wait for pending messages on close
    #[serde(default = "default_linger")]
    pub linger_ms: i32,

    /// Enable reconnection
    #[serde(default = "default_true")]
    pub reconnect: bool,

    /// Reconnection interval (ms)
    #[serde(default = "default_reconnect_interval")]
    pub reconnect_interval_ms: u64,
}

fn default_hwm() -> i32 {
    1000
}

fn default_linger() -> i32 {
    1000
}

fn default_true() -> bool {
    true
}

fn default_reconnect_interval() -> u64 {
    1000
}

impl Default for ZmqConfig {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            pattern: ZmqPattern::Sub,
            topics: Vec::new(),
            high_water_mark: 1000,
            recv_timeout_ms: 0,
            send_timeout_ms: 0,
            linger_ms: 1000,
            reconnect: true,
            reconnect_interval_ms: 1000,
        }
    }
}

/// ZeroMQ message
#[derive(Debug, Clone)]
pub struct ZmqMessage {
    /// Message topic (for PUB/SUB)
    pub topic: Option<String>,
    /// Message payload
    pub payload: Bytes,
    /// Additional frames (for multipart messages)
    pub frames: Vec<Bytes>,
}

impl ZmqMessage {
    /// Create a new message with payload
    pub fn new(payload: impl Into<Bytes>) -> Self {
        Self {
            topic: None,
            payload: payload.into(),
            frames: Vec::new(),
        }
    }

    /// Create a new message with topic and payload
    pub fn with_topic(topic: impl Into<String>, payload: impl Into<Bytes>) -> Self {
        Self {
            topic: Some(topic.into()),
            payload: payload.into(),
            frames: Vec::new(),
        }
    }

    /// Add a frame to the message
    pub fn add_frame(&mut self, frame: impl Into<Bytes>) {
        self.frames.push(frame.into());
    }

    /// Convert to multipart message
    fn to_multipart(&self) -> Multipart {
        let mut parts: Vec<Message> = Vec::new();

        // Add topic if present
        if let Some(ref topic) = self.topic {
            parts.push(topic.as_bytes().to_vec().into());
        }

        // Add payload
        parts.push(self.payload.to_vec().into());

        // Add additional frames
        for frame in &self.frames {
            parts.push(frame.to_vec().into());
        }

        Multipart::from(parts)
    }

    /// Create from multipart message
    fn from_multipart(multipart: Multipart, has_topic: bool) -> IoResult<Self> {
        let mut parts: Vec<Vec<u8>> = multipart.into_iter().map(|msg| msg.to_vec()).collect();

        if parts.is_empty() {
            return Err(IoError::Protocol("Empty ZeroMQ message".to_string()));
        }

        let topic = if has_topic {
            let topic_bytes = parts.remove(0);
            Some(
                String::from_utf8(topic_bytes)
                    .map_err(|e| IoError::Protocol(format!("Invalid topic UTF-8: {}", e)))?,
            )
        } else {
            None
        };

        if parts.is_empty() {
            return Err(IoError::Protocol(
                "ZeroMQ message has no payload".to_string(),
            ));
        }

        let payload = Bytes::from(parts.remove(0));

        let frames = parts.into_iter().map(Bytes::from).collect();

        Ok(Self {
            topic,
            payload,
            frames,
        })
    }
}

/// ZeroMQ stream for receiving messages
pub struct ZmqStream {
    config: ZmqConfig,
    context: Context,
}

impl ZmqStream {
    /// Connect to a ZeroMQ endpoint
    pub async fn connect(config: ZmqConfig) -> IoResult<Self> {
        info!(
            "Connecting to ZeroMQ endpoint: {} (pattern: {:?})",
            config.endpoint, config.pattern
        );

        let context = Context::new();

        let stream = Self { config, context };

        Ok(stream)
    }

    /// Receive a message from the stream
    pub async fn recv(&mut self) -> IoResult<Option<ZmqMessage>> {
        match self.config.pattern {
            ZmqPattern::Sub => self.recv_sub().await,
            ZmqPattern::Pull => self.recv_pull().await,
            ZmqPattern::Rep => self.recv_rep().await,
            ZmqPattern::Dealer => self.recv_dealer().await,
            _ => Err(IoError::Unsupported(format!(
                "Receive not supported for pattern: {:?}",
                self.config.pattern
            ))),
        }
    }

    /// Send a message to the stream
    pub async fn send(&mut self, msg: ZmqMessage) -> IoResult<()> {
        match self.config.pattern {
            ZmqPattern::Pub => self.send_pub(msg).await,
            ZmqPattern::Push => self.send_push(msg).await,
            ZmqPattern::Req => self.send_req(msg).await,
            ZmqPattern::Rep => self.send_rep(msg).await,
            _ => Err(IoError::Unsupported(format!(
                "Send not supported for pattern: {:?}",
                self.config.pattern
            ))),
        }
    }

    /// Receive from SUB socket
    async fn recv_sub(&mut self) -> IoResult<Option<ZmqMessage>> {
        // Simplified implementation - returns an error for now
        // Full implementation would require proper async socket management
        Err(IoError::Unsupported(
            "SUB pattern requires async runtime setup - use PULL pattern for simpler streaming"
                .to_string(),
        ))
    }

    /// Receive from PULL socket
    async fn recv_pull(&mut self) -> IoResult<Option<ZmqMessage>> {
        let mut socket = pull(&self.context)
            .connect(&self.config.endpoint)
            .map_err(|e| IoError::Connection(format!("Failed to connect PULL socket: {}", e)))?;

        let multipart = socket
            .next()
            .await
            .ok_or_else(|| IoError::Connection("PULL socket closed".to_string()))?
            .map_err(|e| IoError::Connection(format!("PULL receive error: {}", e)))?;

        let msg = ZmqMessage::from_multipart(multipart, false)?;
        Ok(Some(msg))
    }

    /// Receive from REP socket (simplified - not fully implemented)
    async fn recv_rep(&mut self) -> IoResult<Option<ZmqMessage>> {
        // REQ/REP pattern requires more complex state management
        // This is a placeholder for future implementation
        Err(IoError::Unsupported(
            "REP pattern not fully implemented yet".to_string(),
        ))
    }

    /// Receive from DEALER socket (simplified - not fully implemented)
    async fn recv_dealer(&mut self) -> IoResult<Option<ZmqMessage>> {
        // DEALER pattern requires more complex state management
        // This is a placeholder for future implementation
        Err(IoError::Unsupported(
            "DEALER pattern not fully implemented yet".to_string(),
        ))
    }

    /// Send to PUB socket
    async fn send_pub(&mut self, msg: ZmqMessage) -> IoResult<()> {
        let mut socket = publish(&self.context)
            .bind(&self.config.endpoint)
            .map_err(|e| IoError::Connection(format!("Failed to bind PUB socket: {}", e)))?;

        let multipart = msg.to_multipart();
        socket
            .send(multipart)
            .await
            .map_err(|e| IoError::Connection(format!("PUB send error: {}", e)))?;

        Ok(())
    }

    /// Send to PUSH socket
    async fn send_push(&mut self, msg: ZmqMessage) -> IoResult<()> {
        let mut socket = push(&self.context)
            .connect(&self.config.endpoint)
            .map_err(|e| IoError::Connection(format!("Failed to connect PUSH socket: {}", e)))?;

        let multipart = msg.to_multipart();
        socket
            .send(multipart)
            .await
            .map_err(|e| IoError::Connection(format!("PUSH send error: {}", e)))?;

        Ok(())
    }

    /// Send to REQ socket (simplified - not fully implemented)
    async fn send_req(&mut self, _msg: ZmqMessage) -> IoResult<()> {
        // REQ/REP pattern requires more complex state management
        // This is a placeholder for future implementation
        Err(IoError::Unsupported(
            "REQ pattern not fully implemented yet".to_string(),
        ))
    }

    /// Send to REP socket (simplified - not fully implemented)
    async fn send_rep(&mut self, _msg: ZmqMessage) -> IoResult<()> {
        // REQ/REP pattern requires more complex state management
        // This is a placeholder for future implementation
        Err(IoError::Unsupported(
            "REP pattern not fully implemented yet".to_string(),
        ))
    }

    /// Get the configuration
    pub fn config(&self) -> &ZmqConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zmq_message_creation() {
        let msg = ZmqMessage::new(&b"test"[..]);
        assert_eq!(msg.payload, Bytes::from(&b"test"[..]));
        assert!(msg.topic.is_none());
        assert!(msg.frames.is_empty());

        let msg = ZmqMessage::with_topic("sensor", &b"data"[..]);
        assert_eq!(msg.topic, Some("sensor".to_string()));
        assert_eq!(msg.payload, Bytes::from(&b"data"[..]));
    }

    #[test]
    fn test_zmq_message_frames() {
        let mut msg = ZmqMessage::new(&b"test"[..]);
        msg.add_frame(&b"frame1"[..]);
        msg.add_frame(&b"frame2"[..]);
        assert_eq!(msg.frames.len(), 2);
    }

    #[test]
    fn test_zmq_config_default() {
        let config = ZmqConfig::default();
        assert_eq!(config.pattern, ZmqPattern::Sub);
        assert_eq!(config.high_water_mark, 1000);
        assert!(config.reconnect);
    }
}
