//! WebSocket stream support
//!
//! Provides WebSocket client for streaming data from WebSocket servers.
//!
//! ## Features
//! - Text and binary message support
//! - Auto-reconnection with exponential backoff
//! - Ping/pong keepalive
//! - Message buffering: `fill_buffer()` pulls a batch of messages off the
//!   network ahead of time into a local `config.buffer_size`-capacity
//!   queue; `try_recv()`/`buffered_len()` give non-blocking access to it,
//!   and `next()` drains it before touching the network.
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
    /// Local FIFO of already-received data messages, sized from
    /// `config.buffer_size`. A previous version allocated this but never
    /// pushed or popped it (`#[allow(dead_code)]`) -- "Message buffering"
    /// was dead code. `fill_buffer()` is the only way messages enter it
    /// (from the network); `next()` drains it before touching the network,
    /// and `try_recv()`/`buffered_len()` give callers non-blocking access
    /// to whatever `fill_buffer()` already pulled in. An earlier attempt at
    /// this fix pushed each message and immediately popped it back out
    /// inside `next()` itself, which touched the field but left it
    /// permanently empty from any caller's perspective -- buffering in
    /// name only.
    buffer: crossbeam_queue::ArrayQueue<Bytes>,
    reconnect_delay: Duration,
}

impl WebSocketStream {
    /// Connect to WebSocket server
    pub async fn connect(config: WebSocketConfig) -> IoResult<Self> {
        let ws = Self::try_connect(&config.url).await?;
        // ArrayQueue::new panics on capacity 0; a deserialized config with
        // buffer_size: 0 must not crash connect().
        let buffer = crossbeam_queue::ArrayQueue::new(config.buffer_size.max(1));
        // Use the configured delay from the start instead of a hardcoded
        // 1000ms literal: a caller setting reconnect_delay_ms = 100 used to
        // still wait a full second before the FIRST reconnect attempt (the
        // configured value was only adopted after a reconnect succeeded).
        let reconnect_delay = Duration::from_millis(config.reconnect_delay_ms);

        info!("WebSocket connected to {}", config.url);

        Ok(Self {
            config,
            ws: Some(ws),
            buffer,
            reconnect_delay,
        })
    }

    /// Non-blocking receive: returns a message already sitting in the
    /// local buffer, if any, without touching the network.
    pub fn try_recv(&self) -> Option<Bytes> {
        self.buffer.pop()
    }

    /// Number of messages currently held in the local buffer.
    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
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

    /// Receive next message (with auto-reconnection). Anything already
    /// sitting in the local buffer (from a prior `fill_buffer()` call) is
    /// returned first, without touching the network.
    pub async fn next(&mut self) -> Option<Bytes> {
        if let Some(buffered) = self.buffer.pop() {
            return Some(buffered);
        }

        loop {
            if self.ws.is_none() {
                // Try to reconnect
                if self.reconnect().await.is_err() {
                    return None;
                }
            }

            let ws = self.ws.as_mut()?;
            let incoming = ws.next().await;

            if let Some(bytes) = self.handle_incoming(incoming).await {
                return Some(bytes);
            }
            // A control frame (ping/pong/close) or a decode error was
            // handled by `handle_incoming`; loop around (possibly
            // reconnecting first, if it cleared `self.ws`).
        }
    }

    /// Eagerly pull up to `max` additional data (Text/Binary) messages from
    /// the network into the local buffer -- replying to any ping and
    /// handling close/error control frames encountered along the way --
    /// and return how many were buffered. Stops early once the buffer's
    /// capacity (from `config.buffer_size`) is reached, the connection ends
    /// and cannot be reconnected, or `max` is reached.
    ///
    /// This is what makes `WebSocketConfig::buffer_size` and the module
    /// doc's "Message buffering" genuinely do something: a caller can pull
    /// a batch off the wire ahead of time via `fill_buffer()` and then
    /// drain it through `try_recv()` (or `next()`, which checks the buffer
    /// first) without a further `.await` per message.
    pub async fn fill_buffer(&mut self, max: usize) -> IoResult<usize> {
        let mut filled = 0usize;

        while filled < max && self.buffer.len() < self.buffer.capacity() {
            if self.ws.is_none() && self.reconnect().await.is_err() {
                if filled == 0 {
                    return Err(IoError::ConnectionFailed(
                        "fill_buffer: not connected and reconnect failed or is disabled".into(),
                    ));
                }
                break;
            }

            let Some(ws) = self.ws.as_mut() else { break };
            let incoming = ws.next().await;

            if let Some(bytes) = self.handle_incoming(incoming).await {
                // Loop condition just checked `buffer.len() < capacity()`
                // and `self` has a single owner (no concurrent pusher), so
                // this push cannot fail.
                if self.buffer.push(bytes).is_ok() {
                    filled += 1;
                }
            }
        }

        Ok(filled)
    }

    /// Process one polled WebSocket protocol frame: replies to pings,
    /// clears `self.ws` on close/error/stream-end, and returns `Some(bytes)`
    /// only for a Text/Binary payload the caller should surface or buffer.
    /// Shared by `next()`'s network path and `fill_buffer()` so the two
    /// cannot drift out of sync on how a given frame type is handled.
    async fn handle_incoming(
        &mut self,
        incoming: Option<Result<Message, tokio_tungstenite::tungstenite::Error>>,
    ) -> Option<Bytes> {
        match incoming {
            Some(Ok(Message::Text(text))) => {
                debug!("Received text: {} bytes", text.len());
                Some(Bytes::from(text.as_bytes().to_vec()))
            }
            Some(Ok(Message::Binary(data))) => {
                debug!("Received binary: {} bytes", data.len());
                Some(data)
            }
            Some(Ok(Message::Ping(data))) => {
                debug!("Received ping");
                if let Some(ws) = self.ws.as_mut() {
                    if let Err(e) = ws.send(Message::Pong(data)).await {
                        error!("Failed to send pong: {}", e);
                        self.ws = None;
                    }
                }
                None
            }
            Some(Ok(Message::Pong(_))) => {
                debug!("Received pong");
                None
            }
            Some(Ok(Message::Close(frame))) => {
                info!("WebSocket closed: {:?}", frame);
                self.ws = None;
                None
            }
            Some(Ok(Message::Frame(_))) => {
                // Raw frames are not exposed in normal operation
                None
            }
            Some(Err(e)) => {
                error!("WebSocket error: {}", e);
                self.ws = None;
                None
            }
            None => {
                warn!("WebSocket stream ended");
                self.ws = None;
                None
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
    pub fn start_ping_task(
        &self,
    ) -> (tokio::task::JoinHandle<()>, tokio::sync::mpsc::Receiver<()>) {
        let url = self.config.url.clone();
        let ping_interval = Duration::from_millis(self.config.ping_interval_ms);

        let (ping_tx, ping_rx) = tokio::sync::mpsc::channel::<()>(16);

        let handle = tokio::spawn(async move {
            if ping_interval.as_millis() == 0 {
                return;
            }

            let mut ticker = interval(ping_interval);
            // skip the immediate first tick
            ticker.tick().await;
            loop {
                ticker.tick().await;
                debug!("Ping interval elapsed for {}", url);
                if ping_tx.send(()).await.is_err() {
                    // Receiver dropped, stop task
                    break;
                }
            }
        });

        (handle, ping_rx)
    }

    /// Drain pending ping signals and send actual WebSocket pings
    pub async fn check_ping_signal(
        &mut self,
        ping_rx: &mut tokio::sync::mpsc::Receiver<()>,
    ) -> IoResult<()> {
        while let Ok(()) = ping_rx.try_recv() {
            self.ping().await?;
        }
        Ok(())
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

        let json = serde_json::to_string(&config).expect("serialization should succeed");
        let deserialized: WebSocketConfig =
            serde_json::from_str(&json).expect("deserialization should succeed");

        assert_eq!(deserialized.url, config.url);
        assert_eq!(deserialized.reconnect, config.reconnect);
    }

    #[cfg(feature = "websocket")]
    #[tokio::test]
    async fn test_ping_task_sends_real_ping() {
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;
        use tokio_tungstenite::tungstenite::protocol::Message as TtMessage;

        // Bind to port 0 — OS assigns a free port
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("should bind to local port");
        let addr = listener.local_addr().expect("should have local addr");

        // Spawn server task
        let server_handle = tokio::spawn(async move {
            let (tcp_stream, _peer) = listener.accept().await.expect("should accept connection");
            let mut ws_server = accept_async(tcp_stream)
                .await
                .expect("WebSocket handshake should succeed");

            // Read one message from the client and return it
            ws_server
                .next()
                .await
                .expect("should receive a message")
                .expect("message should not be an error")
        });

        // Connect client
        let ws_url = format!("ws://{}", addr);
        let config = WebSocketConfig {
            url: ws_url,
            ping_interval_ms: 50, // short interval for the test
            reconnect: false,
            ..WebSocketConfig::default()
        };
        let mut client = WebSocketStream::connect(config)
            .await
            .expect("client should connect");

        // Start ping task (50 ms interval)
        let (_handle, mut ping_rx) = client.start_ping_task();

        // Wait for slightly more than one tick so the channel has a signal
        tokio::time::sleep(Duration::from_millis(120)).await;

        // Drain the signal and send the actual ping
        client
            .check_ping_signal(&mut ping_rx)
            .await
            .expect("check_ping_signal should succeed");

        // Collect what the server saw
        let received = server_handle.await.expect("server task should complete");

        assert!(
            matches!(received, TtMessage::Ping(_)),
            "expected Ping message, got {:?}",
            received
        );
    }

    // === Regression tests: message buffering + reconnect delay (medium, id=45) ===
    //
    // A previous fix attempt pushed each received message into the buffer
    // and immediately popped it back out inside `next()`, so the buffer was
    // always empty from any caller's perspective: `buffered_len()` was
    // always 0 and `try_recv()` always `None`. The discriminating property
    // these tests check is that a public API call can leave
    // `buffered_len() > 0`, with those exact messages then coming back out
    // via `try_recv()` in order.

    #[test]
    fn test_try_recv_and_buffered_len_on_empty_stream() {
        // Sanity check the empty state without a live connection by
        // constructing the queue directly.
        let buffer = crossbeam_queue::ArrayQueue::<Bytes>::new(4);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.pop().is_none());
    }

    #[cfg(feature = "websocket")]
    #[tokio::test]
    async fn test_fill_buffer_then_try_recv_returns_messages_in_order() {
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;
        use tokio_tungstenite::tungstenite::protocol::Message as TtMessage;

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("should bind to local port");
        let addr = listener.local_addr().expect("should have local addr");

        let server_handle = tokio::spawn(async move {
            let (tcp_stream, _peer) = listener.accept().await.expect("should accept connection");
            let mut ws_server = accept_async(tcp_stream)
                .await
                .expect("WebSocket handshake should succeed");
            for i in 0..5u8 {
                ws_server
                    .send(TtMessage::Binary(vec![i].into()))
                    .await
                    .expect("send should succeed");
            }
        });

        let config = WebSocketConfig {
            url: format!("ws://{}", addr),
            reconnect: false,
            buffer_size: 16,
            ..WebSocketConfig::default()
        };
        let mut client = WebSocketStream::connect(config)
            .await
            .expect("client should connect");

        let filled = client
            .fill_buffer(5)
            .await
            .expect("fill_buffer should pull all 5 sent messages");
        assert_eq!(filled, 5, "fill_buffer should report exactly 5 buffered");
        assert_eq!(
            client.buffered_len(),
            5,
            "messages must actually sit in the buffer, not vanish like a previous fix attempt"
        );

        for i in 0..5u8 {
            let msg = client
                .try_recv()
                .expect("a buffered message should be available without touching the network");
            assert_eq!(
                msg.as_ref(),
                &[i],
                "buffered messages must drain in FIFO order"
            );
        }
        assert_eq!(client.buffered_len(), 0);
        assert!(client.try_recv().is_none());

        server_handle.await.expect("server task should complete");
    }

    #[cfg(feature = "websocket")]
    #[tokio::test]
    async fn test_fill_buffer_errors_when_disconnected_and_reconnect_disabled() {
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("should bind to local port");
        let addr = listener.local_addr().expect("should have local addr");

        let server_handle = tokio::spawn(async move {
            let (tcp_stream, _peer) = listener.accept().await.expect("should accept connection");
            let mut ws_server = accept_async(tcp_stream)
                .await
                .expect("WebSocket handshake should succeed");
            // Immediately close from the server side.
            let _ = ws_server.close(None).await;
        });

        let config = WebSocketConfig {
            url: format!("ws://{}", addr),
            reconnect: false,
            ..WebSocketConfig::default()
        };
        let mut client = WebSocketStream::connect(config)
            .await
            .expect("client should connect");

        // The server closes right away and reconnect is disabled, so
        // fill_buffer must surface a real error (not silently claim success
        // with 0 messages, and not hang).
        let result = client.fill_buffer(5).await;
        assert!(
            matches!(result, Err(IoError::ConnectionFailed(_))),
            "expected a ConnectionFailed error, got {result:?}"
        );

        let _ = server_handle.await;
    }

    #[cfg(feature = "websocket")]
    #[tokio::test]
    async fn test_next_drains_buffer_before_network() {
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;
        use tokio_tungstenite::tungstenite::protocol::Message as TtMessage;

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("should bind to local port");
        let addr = listener.local_addr().expect("should have local addr");

        let (more_tx, more_rx) = tokio::sync::oneshot::channel::<()>();
        let server_handle = tokio::spawn(async move {
            let (tcp_stream, _peer) = listener.accept().await.expect("should accept connection");
            let mut ws_server = accept_async(tcp_stream)
                .await
                .expect("WebSocket handshake should succeed");
            for i in 0..3u8 {
                ws_server
                    .send(TtMessage::Binary(vec![i].into()))
                    .await
                    .expect("send should succeed");
            }
            let _ = more_rx.await;
            for i in 3..5u8 {
                ws_server
                    .send(TtMessage::Binary(vec![i].into()))
                    .await
                    .expect("send should succeed");
            }
        });

        let config = WebSocketConfig {
            url: format!("ws://{}", addr),
            reconnect: false,
            buffer_size: 16,
            ..WebSocketConfig::default()
        };
        let mut client = WebSocketStream::connect(config)
            .await
            .expect("client should connect");

        // Pre-fill the buffer with the first 3 messages, then tell the
        // server to send the remaining 2 over the network.
        let filled = client.fill_buffer(3).await.expect("should fill 3");
        assert_eq!(filled, 3);
        let _ = more_tx.send(());

        // `next()` must return the buffered 3 first (in order), then fall
        // through to the network for the last 2 -- still in order overall.
        for i in 0..5u8 {
            let msg = client.next().await.expect("should receive a message");
            assert_eq!(msg.as_ref(), &[i], "overall order must be preserved");
        }

        server_handle.await.expect("server task should complete");
    }

    #[cfg(feature = "websocket")]
    #[tokio::test]
    async fn test_connect_uses_configured_reconnect_delay_not_hardcoded_1000ms() {
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("should bind to local port");
        let addr = listener.local_addr().expect("should have local addr");

        let server_handle = tokio::spawn(async move {
            let (tcp_stream, _peer) = listener.accept().await.expect("should accept connection");
            let _ = accept_async(tcp_stream).await;
            // Keep the task alive briefly so the handshake completes
            // cleanly on the client side.
            tokio::time::sleep(Duration::from_millis(50)).await;
        });

        let config = WebSocketConfig {
            url: format!("ws://{}", addr),
            reconnect_delay_ms: 250,
            reconnect: false,
            ..WebSocketConfig::default()
        };
        let client = WebSocketStream::connect(config)
            .await
            .expect("client should connect");

        // A previous version always initialised this to a hardcoded
        // Duration::from_millis(1000) regardless of config, only adopting
        // the configured value *after* a successful reconnect.
        assert_eq!(client.reconnect_delay, Duration::from_millis(250));

        let _ = server_handle.await;
    }

    #[cfg(feature = "websocket")]
    #[tokio::test]
    async fn test_connect_with_zero_buffer_size_does_not_panic() {
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("should bind to local port");
        let addr = listener.local_addr().expect("should have local addr");

        let server_handle = tokio::spawn(async move {
            let (tcp_stream, _peer) = listener.accept().await.expect("should accept connection");
            let _ = accept_async(tcp_stream).await;
            tokio::time::sleep(Duration::from_millis(50)).await;
        });

        let config = WebSocketConfig {
            url: format!("ws://{}", addr),
            buffer_size: 0, // ArrayQueue::new(0) panics; connect() must clamp this
            reconnect: false,
            ..WebSocketConfig::default()
        };
        // WebSocketStream doesn't implement Debug (it holds a
        // tokio_tungstenite WsStream), so match instead of `{result:?}`.
        match WebSocketStream::connect(config).await {
            Ok(_) => {}
            Err(e) => panic!("expected connect() to succeed with buffer_size=0, got: {e}"),
        }

        let _ = server_handle.await;
    }

    #[cfg(feature = "websocket")]
    #[tokio::test]
    async fn test_next_delivers_multiple_rapid_messages_in_order() {
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;
        use tokio_tungstenite::tungstenite::protocol::Message as TtMessage;

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("should bind to local port");
        let addr = listener.local_addr().expect("should have local addr");

        let server_handle = tokio::spawn(async move {
            let (tcp_stream, _peer) = listener.accept().await.expect("should accept connection");
            let mut ws_server = accept_async(tcp_stream)
                .await
                .expect("WebSocket handshake should succeed");
            for i in 0..5u8 {
                ws_server
                    .send(TtMessage::Binary(vec![i].into()))
                    .await
                    .expect("send should succeed");
            }
        });

        let config = WebSocketConfig {
            url: format!("ws://{}", addr),
            reconnect: false,
            ..WebSocketConfig::default()
        };
        let mut client = WebSocketStream::connect(config)
            .await
            .expect("client should connect");

        for i in 0..5u8 {
            let msg = client.next().await.expect("should receive a message");
            assert_eq!(msg.as_ref(), &[i], "messages must arrive in FIFO order");
        }

        server_handle.await.expect("server task should complete");
    }
}
