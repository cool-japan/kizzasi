//! WebSocket adapter for real-time streaming inference
//!
//! Provides bidirectional streaming over WebSocket protocol for low-latency inference.

use super::{InferenceMessage, InferenceResponse, NetworkAdapter};
use crate::error::{InferenceError, InferenceResult};
use crate::streaming::StreamingEngine;
use futures::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tokio_tungstenite::{accept_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

/// WebSocket adapter for streaming inference
pub struct WebSocketAdapter {
    /// Address to bind to
    addr: SocketAddr,

    /// Streaming engine
    engine: Arc<RwLock<StreamingEngine>>,

    /// Running state
    running: Arc<RwLock<bool>>,
}

impl WebSocketAdapter {
    /// Create a new WebSocket adapter
    ///
    /// # Arguments
    ///
    /// * `addr` - Socket address to bind to (e.g., "127.0.0.1:8080")
    /// * `engine` - Streaming inference engine
    pub fn new(addr: impl Into<SocketAddr>, engine: StreamingEngine) -> Self {
        Self {
            addr: addr.into(),
            engine: Arc::new(RwLock::new(engine)),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Serve WebSocket connections
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let adapter = WebSocketAdapter::new("127.0.0.1:8080".parse()?, engine);
    /// adapter.serve().await?;
    /// ```
    pub async fn serve(&self) -> InferenceResult<()> {
        let listener = TcpListener::bind(&self.addr)
            .await
            .map_err(|e| InferenceError::NetworkError(e.to_string()))?;

        info!("WebSocket server listening on {}", self.addr);
        *self.running.write().await = true;

        while *self.running.read().await {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    info!("New WebSocket connection from {}", addr);
                    let engine = Arc::clone(&self.engine);
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, engine).await {
                            error!("Connection error from {}: {}", addr, e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to accept connection: {}", e);
                }
            }
        }

        Ok(())
    }
}

impl NetworkAdapter for WebSocketAdapter {
    async fn start(&mut self) -> InferenceResult<()> {
        self.serve().await
    }

    async fn stop(&mut self) -> InferenceResult<()> {
        *self.running.write().await = false;
        info!("WebSocket server stopped");
        Ok(())
    }

    fn is_running(&self) -> bool {
        // Need to block on the async read - this is a limitation of the trait design
        // In production, you might want to use a different approach
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async { *self.running.read().await })
        })
    }
}

/// Handle a single WebSocket connection
async fn handle_connection(
    stream: TcpStream,
    engine: Arc<RwLock<StreamingEngine>>,
) -> InferenceResult<()> {
    let ws_stream = accept_async(stream)
        .await
        .map_err(|e| InferenceError::NetworkError(e.to_string()))?;

    let (mut write, mut read) = ws_stream.split();

    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                debug!("Received text message: {}", text);

                // Parse the request
                let request: InferenceMessage = serde_json::from_str(&text)
                    .map_err(|e| InferenceError::SerializationError(e.to_string()))?;

                // Process the request
                let start = std::time::Instant::now();
                let input = request.to_array();

                let engine_guard = engine.write().await;
                let output = engine_guard
                    .step_async(input)
                    .await
                    .map_err(|e| InferenceError::NetworkError(e.to_string()))?;

                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

                // Create response
                let response = InferenceResponse::new(
                    request.request_id,
                    output.to_vec(),
                    latency_ms,
                    output.len(),
                );

                // Send response
                let response_text = serde_json::to_string(&response)
                    .map_err(|e| InferenceError::SerializationError(e.to_string()))?;

                write
                    .send(Message::Text(response_text.into()))
                    .await
                    .map_err(|e| InferenceError::NetworkError(e.to_string()))?;
            }
            Ok(Message::Binary(data)) => {
                debug!("Received binary message ({} bytes)", data.len());

                // Parse binary format (MessagePack or custom binary protocol)
                #[cfg(feature = "msgpack")]
                {
                    let request: InferenceMessage = rmp_serde::from_slice(&data)
                        .map_err(|e| InferenceError::SerializationError(e.to_string()))?;

                    let start = std::time::Instant::now();
                    let input = request.to_array();

                    let engine_guard = engine.write().await;
                    let output = engine_guard
                        .step_async(input)
                        .await
                        .map_err(|e| InferenceError::NetworkError(e.to_string()))?;

                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

                    let response = InferenceResponse::new(
                        request.request_id,
                        output.to_vec(),
                        latency_ms,
                        output.len(),
                    );

                    let response_data = rmp_serde::to_vec(&response)
                        .map_err(|e| InferenceError::SerializationError(e.to_string()))?;

                    write
                        .send(Message::Binary(response_data.into()))
                        .await
                        .map_err(|e| InferenceError::NetworkError(e.to_string()))?;
                }

                #[cfg(not(feature = "msgpack"))]
                {
                    warn!("Binary messages require msgpack feature");
                }
            }
            Ok(Message::Close(_)) => {
                info!("WebSocket connection closed by client");
                break;
            }
            Ok(Message::Ping(data)) => {
                write
                    .send(Message::Pong(data))
                    .await
                    .map_err(|e| InferenceError::NetworkError(e.to_string()))?;
            }
            Ok(Message::Pong(_)) => {
                // Ignore pongs
            }
            Ok(Message::Frame(_)) => {
                warn!("Received raw frame (unexpected)");
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_websocket_adapter_creation() {
        use crate::streaming::StreamConfig;

        let stream_config = StreamConfig::default();
        let engine = StreamingEngine::new(stream_config).unwrap();

        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let adapter = WebSocketAdapter::new(addr, engine);

        assert!(!adapter.is_running());
    }

    #[test]
    fn test_inference_message() {
        let msg = InferenceMessage::new("test-123", vec![1.0, 2.0, 3.0]);
        assert_eq!(msg.request_id, "test-123");
        assert_eq!(msg.input.len(), 3);

        let arr = msg.to_array();
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_inference_response() {
        let resp = InferenceResponse::new("test-456", vec![4.0, 5.0], 12.5, 2);
        assert_eq!(resp.request_id, "test-456");
        assert_eq!(resp.output.len(), 2);
        assert_eq!(resp.latency_ms, 12.5);
        assert_eq!(resp.num_tokens, 2);

        let arr = resp.to_array();
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_message_serialization() {
        let msg = InferenceMessage::new("req-1", vec![1.0, 2.0, 3.0]);
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: InferenceMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(msg.request_id, deserialized.request_id);
        assert_eq!(msg.input, deserialized.input);
    }

    #[test]
    fn test_response_serialization() {
        let resp = InferenceResponse::new("resp-1", vec![1.0, 2.0], 10.0, 2);
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: InferenceResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(resp.request_id, deserialized.request_id);
        assert_eq!(resp.output, deserialized.output);
    }
}
