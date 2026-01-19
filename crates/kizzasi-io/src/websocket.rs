//! WebSocket stream support
//!
//! Provides WebSocket client for streaming data from WebSocket servers.
//!
//! ## Features
//! - Text and binary message support
//! - Auto-reconnection with exponential backoff
//! - Ping/pong keepalive
//! - Message buffering
//!
//! ## Example
//! ```rust,no_run
//! use kizzasi_io::{WebSocketStream, WebSocketConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = WebSocketConfig {
//!         url: "ws://localhost:8080/stream".to_string(),
//!         ..Default::default()
//!     };
//!
//!     let mut stream = WebSocketStream::connect(config).await?;
//!
//!     while let Some(data) = stream.next().await {
//!         println!("Received: {:?}", data);
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::error::{IoError, IoResult};
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::time::{interval, sleep};
use tokio_tungstenite::{
    connect_async, tungstenite::protocol::Message, MaybeTlsStream, WebSocketStream as WsStream,
};
use tracing::{debug, error, info, warn};

/// WebSocket configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketConfig {
    /// WebSocket server URL (ws:// or wss://)
    pub url: String,

    /// Reconnection enabled
    #[serde(default = "default_true")]
    pub reconnect: bool,

    /// Initial reconnection delay (ms)
    #[serde(default = "default_reconnect_delay")]
    pub reconnect_delay_ms: u64,

    /// Maximum reconnection delay (ms)
    #[serde(default = "default_max_reconnect_delay")]
    pub max_reconnect_delay_ms: u64,

    /// Ping interval (ms) - 0 to disable
    #[serde(default = "default_ping_interval")]
    pub ping_interval_ms: u64,

    /// Message buffer size
    #[serde(default = "default_buffer_size")]
    pub buffer_size: usize,
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

fn default_ping_interval() -> u64 {
    30000
}

fn default_buffer_size() -> usize {
    1024
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            reconnect: true,
            reconnect_delay_ms: 1000,
            max_reconnect_delay_ms: 30000,
            ping_interval_ms: 30000,
            buffer_size: 1024,
        }
    }
}

/// WebSocket stream for real-time data
pub struct WebSocketStream {
    config: WebSocketConfig,
    ws: Option<WsStream<MaybeTlsStream<TcpStream>>>,
    #[allow(dead_code)]
    buffer: crossbeam_queue::ArrayQueue<Bytes>,
    reconnect_delay: Duration,
}

impl WebSocketStream {
    /// Connect to WebSocket server
    pub async fn connect(config: WebSocketConfig) -> IoResult<Self> {
        let ws = Self::try_connect(&config.url).await?;
        let buffer = crossbeam_queue::ArrayQueue::new(config.buffer_size);

        info!("WebSocket connected to {}", config.url);

        Ok(Self {
            config,
            ws: Some(ws),
            buffer,
            reconnect_delay: Duration::from_millis(1000),
        })
    }

    /// Try to connect to WebSocket server
    async fn try_connect(url: &str) -> IoResult<WsStream<MaybeTlsStream<TcpStream>>> {
        let (ws_stream, response) = connect_async(url)
            .await
            .map_err(|e| IoError::ConnectionFailed(format!("WebSocket connect failed: {}", e)))?;

        debug!("WebSocket response: {:?}", response);
        Ok(ws_stream)
    }

    /// Reconnect to WebSocket server with exponential backoff
    async fn reconnect(&mut self) -> IoResult<()> {
        if !self.config.reconnect {
            return Err(IoError::ConnectionFailed("Reconnection disabled".into()));
        }

        warn!(
            "Attempting to reconnect to {} (delay: {:?})",
            self.config.url, self.reconnect_delay
        );

        sleep(self.reconnect_delay).await;

        match Self::try_connect(&self.config.url).await {
            Ok(ws) => {
                self.ws = Some(ws);
                self.reconnect_delay = Duration::from_millis(self.config.reconnect_delay_ms);
                info!("WebSocket reconnected successfully");
                Ok(())
            }
            Err(e) => {
                // Exponential backoff
                let new_delay = self.reconnect_delay.as_millis() * 2;
                let max_delay = self.config.max_reconnect_delay_ms as u128;
                self.reconnect_delay = Duration::from_millis(new_delay.min(max_delay) as u64);

                error!("WebSocket reconnection failed: {}", e);
                Err(e)
            }
        }
    }

    /// Receive next message (with auto-reconnection)
    pub async fn next(&mut self) -> Option<Bytes> {
        loop {
            if self.ws.is_none() {
                // Try to reconnect
                if self.reconnect().await.is_err() {
                    return None;
                }
            }

            let ws = self.ws.as_mut()?;

            match ws.next().await {
                Some(Ok(msg)) => {
                    match msg {
                        Message::Text(text) => {
                            debug!("Received text: {} bytes", text.len());
                            return Some(Bytes::from(text.as_bytes().to_vec()));
                        }
                        Message::Binary(data) => {
                            debug!("Received binary: {} bytes", data.len());
                            return Some(data);
                        }
                        Message::Ping(data) => {
                            debug!("Received ping");
                            if let Err(e) = ws.send(Message::Pong(data)).await {
                                error!("Failed to send pong: {}", e);
                                self.ws = None;
                            }
                        }
                        Message::Pong(_) => {
                            debug!("Received pong");
                        }
                        Message::Close(frame) => {
                            info!("WebSocket closed: {:?}", frame);
                            self.ws = None;
                        }
                        Message::Frame(_) => {
                            // Raw frames are not exposed in normal operation
                        }
                    }
                }
                Some(Err(e)) => {
                    error!("WebSocket error: {}", e);
                    self.ws = None;
                }
                None => {
                    warn!("WebSocket stream ended");
                    self.ws = None;
                }
            }
        }
    }

    /// Send a text message
    pub async fn send_text(&mut self, text: String) -> IoResult<()> {
        let ws = self
            .ws
            .as_mut()
            .ok_or_else(|| IoError::ConnectionFailed("Not connected".into()))?;

        ws.send(Message::Text(text.into()))
            .await
            .map_err(|e| IoError::SendFailed(format!("Failed to send text: {}", e)))
    }

    /// Send a binary message
    pub async fn send_binary(&mut self, data: Vec<u8>) -> IoResult<()> {
        let ws = self
            .ws
            .as_mut()
            .ok_or_else(|| IoError::ConnectionFailed("Not connected".into()))?;

        ws.send(Message::Binary(data.into()))
            .await
            .map_err(|e| IoError::SendFailed(format!("Failed to send binary: {}", e)))
    }

    /// Send ping
    pub async fn ping(&mut self) -> IoResult<()> {
        let ws = self
            .ws
            .as_mut()
            .ok_or_else(|| IoError::ConnectionFailed("Not connected".into()))?;

        ws.send(Message::Ping(vec![].into()))
            .await
            .map_err(|e| IoError::SendFailed(format!("Failed to send ping: {}", e)))
    }

    /// Start ping task
    pub fn start_ping_task(&self) -> tokio::task::JoinHandle<()> {
        let url = self.config.url.clone();
        let ping_interval = Duration::from_millis(self.config.ping_interval_ms);

        tokio::spawn(async move {
            if ping_interval.as_millis() == 0 {
                return;
            }

            let mut ticker = interval(ping_interval);
            loop {
                ticker.tick().await;
                debug!("Ping interval elapsed for {}", url);
                // Note: actual ping sending would need access to the WebSocket
                // This is a placeholder for keepalive logic
            }
        })
    }

    /// Close the WebSocket connection
    pub async fn close(&mut self) -> IoResult<()> {
        if let Some(mut ws) = self.ws.take() {
            ws.close(None)
                .await
                .map_err(|e| IoError::ConnectionFailed(format!("Failed to close: {}", e)))?;
            info!("WebSocket closed");
        }
        Ok(())
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.ws.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = WebSocketConfig::default();
        assert!(config.reconnect);
        assert_eq!(config.reconnect_delay_ms, 1000);
        assert_eq!(config.max_reconnect_delay_ms, 30000);
        assert_eq!(config.ping_interval_ms, 30000);
        assert_eq!(config.buffer_size, 1024);
    }

    #[test]
    fn test_config_serialize() {
        let config = WebSocketConfig {
            url: "ws://localhost:8080".to_string(),
            reconnect: true,
            reconnect_delay_ms: 2000,
            max_reconnect_delay_ms: 60000,
            ping_interval_ms: 15000,
            buffer_size: 2048,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: WebSocketConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.url, config.url);
        assert_eq!(deserialized.reconnect, config.reconnect);
    }
}
