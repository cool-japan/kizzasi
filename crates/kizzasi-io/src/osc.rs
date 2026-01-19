//! OSC (Open Sound Control) protocol support
//!
//! Provides OSC client and server for communication with audio/music software.
//!
//! ## Features
//! - UDP-based OSC communication
//! - Message sending and receiving
//! - Bundle support
//! - Type-safe message construction
//! - Pattern matching
//!
//! ## Example
//! ```rust,no_run
//! use kizzasi_io::{OscSender, OscReceiver, OscMessage};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Send OSC messages
//!     let sender = OscSender::new("127.0.0.1:9000").await?;
//!     sender.send_float("/volume", 0.5).await?;
//!
//!     // Receive OSC messages
//!     let mut receiver = OscReceiver::new("127.0.0.1:8000").await?;
//!     while let Some(msg) = receiver.recv().await? {
//!         println!("Received: {:?}", msg);
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::error::{IoError, IoResult};
use rosc::{OscBundle, OscMessage as RoscMessage, OscPacket, OscType};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tokio::net::UdpSocket;
use tracing::{debug, info, warn};

/// Type alias for OSC message handler functions
type OscHandler = Box<dyn Fn(&OscMessage) + Send + Sync>;

/// OSC message with address and arguments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscMessage {
    /// OSC address pattern (e.g., "/synth/volume")
    pub address: String,

    /// Arguments
    pub args: Vec<OscArg>,
}

/// OSC argument types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OscArg {
    /// 32-bit integer
    Int(i32),
    /// 32-bit float
    Float(f32),
    /// String
    String(String),
    /// Blob (binary data)
    Blob(Vec<u8>),
    /// 64-bit integer
    Long(i64),
    /// 64-bit float
    Double(f64),
    /// Boolean
    Bool(bool),
    /// Nil (null)
    Nil,
    /// Impulse (bang)
    Impulse,
}

impl From<OscType> for OscArg {
    fn from(osc_type: OscType) -> Self {
        match osc_type {
            OscType::Int(v) => OscArg::Int(v),
            OscType::Float(v) => OscArg::Float(v),
            OscType::String(v) => OscArg::String(v),
            OscType::Blob(v) => OscArg::Blob(v),
            OscType::Long(v) => OscArg::Long(v),
            OscType::Double(v) => OscArg::Double(v),
            OscType::Bool(v) => OscArg::Bool(v),
            OscType::Nil => OscArg::Nil,
            OscType::Inf => OscArg::Double(f64::INFINITY),
            OscType::Time(_) => OscArg::Long(0), // Convert time tag to long
            OscType::Char(c) => OscArg::String(c.to_string()),
            OscType::Color(_) => OscArg::Int(0), // Convert color to int
            OscType::Midi(_) => OscArg::Blob(vec![]), // Convert MIDI to blob
            OscType::Array(arr) => {
                // Take first element or default to nil
                arr.content
                    .into_iter()
                    .next()
                    .map(|t| t.into())
                    .unwrap_or(OscArg::Nil)
            }
        }
    }
}

impl From<OscArg> for OscType {
    fn from(arg: OscArg) -> Self {
        match arg {
            OscArg::Int(v) => OscType::Int(v),
            OscArg::Float(v) => OscType::Float(v),
            OscArg::String(v) => OscType::String(v),
            OscArg::Blob(v) => OscType::Blob(v),
            OscArg::Long(v) => OscType::Long(v),
            OscArg::Double(v) => OscType::Double(v),
            OscArg::Bool(v) => OscType::Bool(v),
            OscArg::Nil => OscType::Nil,
            OscArg::Impulse => OscType::Inf,
        }
    }
}

/// OSC sender for sending messages
pub struct OscSender {
    socket: UdpSocket,
    target: SocketAddr,
}

impl OscSender {
    /// Create a new OSC sender
    pub async fn new(target: &str) -> IoResult<Self> {
        let target_addr: SocketAddr = target
            .parse()
            .map_err(|e| IoError::ConfigError(format!("Invalid target address: {}", e)))?;

        let socket = UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| IoError::ConnectionFailed(format!("Failed to bind UDP socket: {}", e)))?;

        info!("OSC sender created, target: {}", target_addr);

        Ok(Self {
            socket,
            target: target_addr,
        })
    }

    /// Send an OSC message
    pub async fn send(&self, message: &OscMessage) -> IoResult<()> {
        let rosc_msg = RoscMessage {
            addr: message.address.clone(),
            args: message.args.iter().map(|a| a.clone().into()).collect(),
        };

        let packet = OscPacket::Message(rosc_msg);
        let encoded = rosc::encoder::encode(&packet)
            .map_err(|e| IoError::SendFailed(format!("Failed to encode OSC message: {}", e)))?;

        self.socket
            .send_to(&encoded, self.target)
            .await
            .map_err(|e| IoError::SendFailed(format!("Failed to send OSC message: {}", e)))?;

        debug!("Sent OSC message to {}: {}", self.target, message.address);

        Ok(())
    }

    /// Send a simple float message
    pub async fn send_float(&self, address: &str, value: f32) -> IoResult<()> {
        let message = OscMessage {
            address: address.to_string(),
            args: vec![OscArg::Float(value)],
        };
        self.send(&message).await
    }

    /// Send a simple int message
    pub async fn send_int(&self, address: &str, value: i32) -> IoResult<()> {
        let message = OscMessage {
            address: address.to_string(),
            args: vec![OscArg::Int(value)],
        };
        self.send(&message).await
    }

    /// Send a simple string message
    pub async fn send_string(&self, address: &str, value: &str) -> IoResult<()> {
        let message = OscMessage {
            address: address.to_string(),
            args: vec![OscArg::String(value.to_string())],
        };
        self.send(&message).await
    }

    /// Send multiple values
    pub async fn send_values(&self, address: &str, args: Vec<OscArg>) -> IoResult<()> {
        let message = OscMessage {
            address: address.to_string(),
            args,
        };
        self.send(&message).await
    }

    /// Send an OSC bundle (multiple messages with timestamp)
    pub async fn send_bundle(&self, messages: Vec<OscMessage>) -> IoResult<()> {
        let rosc_messages: Vec<OscPacket> = messages
            .into_iter()
            .map(|msg| {
                OscPacket::Message(RoscMessage {
                    addr: msg.address,
                    args: msg.args.into_iter().map(|a| a.into()).collect(),
                })
            })
            .collect();

        let bundle = OscBundle {
            timetag: (0, 0).into(), // Immediate
            content: rosc_messages,
        };

        let packet = OscPacket::Bundle(bundle);
        let encoded = rosc::encoder::encode(&packet)
            .map_err(|e| IoError::SendFailed(format!("Failed to encode OSC bundle: {}", e)))?;

        self.socket
            .send_to(&encoded, self.target)
            .await
            .map_err(|e| IoError::SendFailed(format!("Failed to send OSC bundle: {}", e)))?;

        debug!("Sent OSC bundle to {}", self.target);

        Ok(())
    }
}

/// OSC receiver for receiving messages
pub struct OscReceiver {
    socket: UdpSocket,
    buffer: Vec<u8>,
}

impl OscReceiver {
    /// Create a new OSC receiver
    pub async fn new(bind_addr: &str) -> IoResult<Self> {
        let socket = UdpSocket::bind(bind_addr)
            .await
            .map_err(|e| IoError::ConnectionFailed(format!("Failed to bind UDP socket: {}", e)))?;

        let local_addr = socket.local_addr().map_err(|e| {
            IoError::ConnectionFailed(format!("Failed to get local address: {}", e))
        })?;

        info!("OSC receiver listening on {}", local_addr);

        Ok(Self {
            socket,
            buffer: vec![0u8; 65536], // 64KB buffer
        })
    }

    /// Receive an OSC message
    pub async fn recv(&mut self) -> IoResult<Option<OscMessage>> {
        let (size, _addr) =
            self.socket.recv_from(&mut self.buffer).await.map_err(|e| {
                IoError::ReadFailed(format!("Failed to receive OSC message: {}", e))
            })?;

        let packet = rosc::decoder::decode_udp(&self.buffer[..size])
            .map_err(|e| IoError::ReadFailed(format!("Failed to decode OSC packet: {}", e)))?;

        match packet.1 {
            OscPacket::Message(msg) => {
                debug!("Received OSC message: {}", msg.addr);

                Ok(Some(OscMessage {
                    address: msg.addr,
                    args: msg.args.into_iter().map(|a| a.into()).collect(),
                }))
            }
            OscPacket::Bundle(bundle) => {
                debug!("Received OSC bundle with {} messages", bundle.content.len());

                // Return first message from bundle
                if let Some(OscPacket::Message(msg)) = bundle.content.into_iter().next() {
                    Ok(Some(OscMessage {
                        address: msg.addr,
                        args: msg.args.into_iter().map(|a| a.into()).collect(),
                    }))
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Receive with timeout
    pub async fn recv_timeout(
        &mut self,
        timeout: std::time::Duration,
    ) -> IoResult<Option<OscMessage>> {
        match tokio::time::timeout(timeout, self.recv()).await {
            Ok(result) => result,
            Err(_) => Ok(None),
        }
    }

    /// Get local address
    pub fn local_addr(&self) -> IoResult<SocketAddr> {
        self.socket
            .local_addr()
            .map_err(|e| IoError::ConnectionFailed(format!("Failed to get local address: {}", e)))
    }
}

/// OSC server with pattern matching
pub struct OscServer {
    receiver: OscReceiver,
    handlers: Vec<(String, OscHandler)>,
}

impl OscServer {
    /// Create a new OSC server
    pub async fn new(bind_addr: &str) -> IoResult<Self> {
        let receiver = OscReceiver::new(bind_addr).await?;

        Ok(Self {
            receiver,
            handlers: Vec::new(),
        })
    }

    /// Add a message handler for an address pattern
    pub fn add_handler<F>(&mut self, pattern: &str, handler: F)
    where
        F: Fn(&OscMessage) + Send + Sync + 'static,
    {
        self.handlers.push((pattern.to_string(), Box::new(handler)));
        info!("Added OSC handler for pattern: {}", pattern);
    }

    /// Start the server loop
    pub async fn run(&mut self) -> IoResult<()> {
        info!("OSC server started");

        loop {
            match self.receiver.recv().await? {
                Some(msg) => {
                    // Find matching handlers
                    let mut handled = false;
                    for (pattern, handler) in &self.handlers {
                        if self.matches_pattern(&msg.address, pattern) {
                            handler(&msg);
                            handled = true;
                        }
                    }

                    if !handled {
                        warn!("Unhandled OSC message: {}", msg.address);
                    }
                }
                None => continue,
            }
        }
    }

    /// Simple pattern matching (supports * wildcard)
    fn matches_pattern(&self, address: &str, pattern: &str) -> bool {
        if pattern == "*" {
            return true;
        }

        if pattern.contains('*') {
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 {
                return address.starts_with(parts[0]) && address.ends_with(parts[1]);
            }
        }

        address == pattern
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_osc_message_creation() {
        let msg = OscMessage {
            address: "/test".to_string(),
            args: vec![
                OscArg::Float(1.0),
                OscArg::Int(42),
                OscArg::String("hello".to_string()),
            ],
        };

        assert_eq!(msg.address, "/test");
        assert_eq!(msg.args.len(), 3);
    }

    #[test]
    fn test_osc_arg_conversion() {
        let float_arg = OscArg::Float(std::f32::consts::PI);
        let osc_type: OscType = float_arg.into();
        assert!(matches!(osc_type, OscType::Float(_)));

        let int_arg = OscArg::Int(42);
        let osc_type: OscType = int_arg.into();
        assert!(matches!(osc_type, OscType::Int(42)));
    }

    #[tokio::test]
    async fn test_osc_sender_receiver() {
        let receiver_addr = "127.0.0.1:0";
        let mut receiver = OscReceiver::new(receiver_addr).await.unwrap();
        let actual_addr = receiver.local_addr().unwrap();

        let sender = OscSender::new(&actual_addr.to_string()).await.unwrap();

        // Send in background
        let send_handle = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            sender.send_float("/test", 1.23).await.unwrap();
        });

        // Receive with timeout
        let result = receiver
            .recv_timeout(std::time::Duration::from_secs(2))
            .await
            .unwrap();

        assert!(result.is_some());
        let msg = result.unwrap();
        assert_eq!(msg.address, "/test");

        send_handle.await.unwrap();
    }
}
