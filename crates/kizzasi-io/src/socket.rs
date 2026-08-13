//! TCP/UDP socket stream support
//!
//! Provides TCP and UDP socket streams for network communication.
//!
//! ## Features
//! - TCP client and server streams
//! - UDP datagram streams
//! - Async I/O with tokio
//! - Buffer management
//!
//! ## Example
//! ```rust,no_run
//! use kizzasi_io::{TcpClientStream, SocketConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = SocketConfig {
//!         address: "127.0.0.1:8080".to_string(),
//!         ..Default::default()
//!     };
//!
//!     let mut stream = TcpClientStream::connect(config).await?;
//!
//!     while let Some(data) = stream.recv().await? {
//!         println!("Received: {:?}", data);
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::error::{IoError, IoResult};
use bytes::{Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tracing::{debug, info};

/// Socket configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocketConfig {
    /// Socket address (host:port)
    pub address: String,

    /// Buffer size for reading
    #[serde(default = "default_buffer_size")]
    pub buffer_size: usize,

    /// Read timeout (ms) - 0 for no timeout
    #[serde(default)]
    pub read_timeout_ms: u64,
}

fn default_buffer_size() -> usize {
    8192
}

impl Default for SocketConfig {
    fn default() -> Self {
        Self {
            address: String::new(),
            buffer_size: 8192,
            read_timeout_ms: 0,
        }
    }
}

/// TCP client stream
pub struct TcpClientStream {
    config: SocketConfig,
    stream: TcpStream,
    buffer: BytesMut,
}

impl TcpClientStream {
    /// Connect to TCP server
    pub async fn connect(config: SocketConfig) -> IoResult<Self> {
        let stream = TcpStream::connect(&config.address)
            .await
            .map_err(|e| IoError::ConnectionFailed(format!("TCP connect failed: {}", e)))?;

        info!("TCP connected to {}", config.address);

        Ok(Self {
            buffer: BytesMut::with_capacity(config.buffer_size),
            config,
            stream,
        })
    }

    /// Receive data from TCP stream
    pub async fn recv(&mut self) -> IoResult<Option<Bytes>> {
        self.buffer.resize(self.config.buffer_size, 0);

        match self.stream.read(&mut self.buffer).await {
            Ok(0) => {
                info!("TCP connection closed");
                Ok(None)
            }
            Ok(n) => {
                debug!("TCP received {} bytes", n);
                Ok(Some(self.buffer[..n].to_vec().into()))
            }
            Err(e) => Err(IoError::ReadFailed(format!("TCP read error: {}", e))),
        }
    }

    /// Send data to TCP stream
    pub async fn send(&mut self, data: &[u8]) -> IoResult<()> {
        self.stream
            .write_all(data)
            .await
            .map_err(|e| IoError::SendFailed(format!("TCP send error: {}", e)))?;

        debug!("TCP sent {} bytes", data.len());
        Ok(())
    }

    /// Get peer address
    pub fn peer_addr(&self) -> IoResult<SocketAddr> {
        self.stream
            .peer_addr()
            .map_err(|e| IoError::ConnectionFailed(format!("Failed to get peer addr: {}", e)))
    }

    /// Get local address
    pub fn local_addr(&self) -> IoResult<SocketAddr> {
        self.stream
            .local_addr()
            .map_err(|e| IoError::ConnectionFailed(format!("Failed to get local addr: {}", e)))
    }

    /// Shutdown the TCP connection
    pub async fn shutdown(&mut self) -> IoResult<()> {
        self.stream
            .shutdown()
            .await
            .map_err(|e| IoError::ConnectionFailed(format!("TCP shutdown error: {}", e)))?;

        info!("TCP connection shutdown");
        Ok(())
    }
}

/// TCP server stream
pub struct TcpServerStream {
    config: SocketConfig,
    listener: TcpListener,
}

impl TcpServerStream {
    /// Bind TCP server
    pub async fn bind(config: SocketConfig) -> IoResult<Self> {
        let listener = TcpListener::bind(&config.address)
            .await
            .map_err(|e| IoError::ConnectionFailed(format!("TCP bind failed: {}", e)))?;

        info!("TCP server listening on {}", config.address);

        Ok(Self { config, listener })
    }

    /// Accept incoming TCP connection
    pub async fn accept(&self) -> IoResult<TcpClientStream> {
        let (stream, addr) = self
            .listener
            .accept()
            .await
            .map_err(|e| IoError::ConnectionFailed(format!("TCP accept failed: {}", e)))?;

        info!("TCP accepted connection from {}", addr);

        Ok(TcpClientStream {
            config: self.config.clone(),
            stream,
            buffer: BytesMut::with_capacity(self.config.buffer_size),
        })
    }

    /// Get local address
    pub fn local_addr(&self) -> IoResult<SocketAddr> {
        self.listener
            .local_addr()
            .map_err(|e| IoError::ConnectionFailed(format!("Failed to get local addr: {}", e)))
    }
}

/// UDP socket stream
pub struct UdpSocketStream {
    config: SocketConfig,
    socket: UdpSocket,
    buffer: BytesMut,
}

impl UdpSocketStream {
    /// Bind UDP socket
    pub async fn bind(config: SocketConfig) -> IoResult<Self> {
        let socket = UdpSocket::bind(&config.address)
            .await
            .map_err(|e| IoError::ConnectionFailed(format!("UDP bind failed: {}", e)))?;

        info!("UDP socket bound to {}", config.address);

        Ok(Self {
            buffer: BytesMut::with_capacity(config.buffer_size),
            config,
            socket,
        })
    }

    /// Connect UDP socket to remote address
    pub async fn connect(config: SocketConfig) -> IoResult<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| IoError::ConnectionFailed(format!("UDP bind failed: {}", e)))?;

        socket
            .connect(&config.address)
            .await
            .map_err(|e| IoError::ConnectionFailed(format!("UDP connect failed: {}", e)))?;

        info!("UDP socket connected to {}", config.address);

        Ok(Self {
            buffer: BytesMut::with_capacity(config.buffer_size),
            config,
            socket,
        })
    }

    /// Receive datagram
    pub async fn recv(&mut self) -> IoResult<(Bytes, SocketAddr)> {
        self.buffer.resize(self.config.buffer_size, 0);

        match self.socket.recv_from(&mut self.buffer).await {
            Ok((n, addr)) => {
                debug!("UDP received {} bytes from {}", n, addr);
                Ok((self.buffer[..n].to_vec().into(), addr))
            }
            Err(e) => Err(IoError::ReadFailed(format!("UDP recv error: {}", e))),
        }
    }

    /// Send datagram
    pub async fn send(&self, data: &[u8]) -> IoResult<()> {
        self.socket
            .send(data)
            .await
            .map_err(|e| IoError::SendFailed(format!("UDP send error: {}", e)))?;

        debug!("UDP sent {} bytes", data.len());
        Ok(())
    }

    /// Send datagram to specific address
    pub async fn send_to(&self, data: &[u8], addr: &str) -> IoResult<()> {
        self.socket
            .send_to(data, addr)
            .await
            .map_err(|e| IoError::SendFailed(format!("UDP send_to error: {}", e)))?;

        debug!("UDP sent {} bytes to {}", data.len(), addr);
        Ok(())
    }

    /// Get local address
    pub fn local_addr(&self) -> IoResult<SocketAddr> {
        self.socket
            .local_addr()
            .map_err(|e| IoError::ConnectionFailed(format!("Failed to get local addr: {}", e)))
    }

    /// Set broadcast mode
    pub fn set_broadcast(&self, broadcast: bool) -> IoResult<()> {
        self.socket
            .set_broadcast(broadcast)
            .map_err(|e| IoError::ConfigError(format!("Failed to set broadcast: {}", e)))
    }

    /// Set multicast loop
    pub fn set_multicast_loop_v4(&self, multicast_loop: bool) -> IoResult<()> {
        self.socket
            .set_multicast_loop_v4(multicast_loop)
            .map_err(|e| IoError::ConfigError(format!("Failed to set multicast loop: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_socket_config_defaults() {
        let config = SocketConfig::default();
        assert_eq!(config.buffer_size, 8192);
        assert_eq!(config.read_timeout_ms, 0);
    }

    #[test]
    fn test_socket_config_serialize() {
        let config = SocketConfig {
            address: "127.0.0.1:8080".to_string(),
            buffer_size: 4096,
            read_timeout_ms: 5000,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SocketConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.address, config.address);
        assert_eq!(deserialized.buffer_size, config.buffer_size);
        assert_eq!(deserialized.read_timeout_ms, config.read_timeout_ms);
    }

    #[tokio::test]
    async fn test_tcp_server_bind() {
        let config = SocketConfig {
            address: "127.0.0.1:0".to_string(),
            ..Default::default()
        };

        let server = TcpServerStream::bind(config).await.unwrap();
        assert!(server.local_addr().is_ok());
    }

    #[tokio::test]
    async fn test_udp_socket_bind() {
        let config = SocketConfig {
            address: "127.0.0.1:0".to_string(),
            ..Default::default()
        };

        let socket = UdpSocketStream::bind(config).await.unwrap();
        assert!(socket.local_addr().is_ok());
    }

    // === Loopback data-path tests (test-gap, id=49) ===
    //
    // The suite previously only constructed config structs and checked that
    // a socket could bind; nothing exercised send/recv, so a broken data
    // path would have passed.

    #[tokio::test]
    async fn test_tcp_loopback_round_trip() {
        let server = TcpServerStream::bind(SocketConfig {
            address: "127.0.0.1:0".to_string(),
            ..Default::default()
        })
        .await
        .unwrap();
        let address = server.local_addr().unwrap().to_string();

        let accept = tokio::spawn(async move {
            let mut accepted = server.accept().await.unwrap();
            let received = accepted.recv().await.unwrap().expect("client sent data");
            // Echo it straight back.
            accepted.send(&received).await.unwrap();
            received
        });

        let mut client = TcpClientStream::connect(SocketConfig {
            address: address.clone(),
            ..Default::default()
        })
        .await
        .unwrap();
        assert_eq!(client.peer_addr().unwrap().to_string(), address);

        let payload = b"kizzasi-io tcp loopback";
        client.send(payload).await.unwrap();

        let echoed = client.recv().await.unwrap().expect("server echoed data");
        assert_eq!(echoed.as_ref(), payload);
        assert_eq!(accept.await.unwrap().as_ref(), payload);

        client.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_tcp_recv_reports_closed_connection() {
        let server = TcpServerStream::bind(SocketConfig {
            address: "127.0.0.1:0".to_string(),
            ..Default::default()
        })
        .await
        .unwrap();
        let address = server.local_addr().unwrap().to_string();

        let accept = tokio::spawn(async move {
            let mut accepted = server.accept().await.unwrap();
            accepted.shutdown().await.unwrap();
        });

        let mut client = TcpClientStream::connect(SocketConfig {
            address,
            ..Default::default()
        })
        .await
        .unwrap();

        // A clean close is reported as `Ok(None)`, not as an error.
        assert!(client.recv().await.unwrap().is_none());
        accept.await.unwrap();
    }

    #[tokio::test]
    async fn test_udp_loopback_round_trip() {
        let mut receiver = UdpSocketStream::bind(SocketConfig {
            address: "127.0.0.1:0".to_string(),
            ..Default::default()
        })
        .await
        .unwrap();
        let address = receiver.local_addr().unwrap().to_string();

        let sender = UdpSocketStream::connect(SocketConfig {
            address: address.clone(),
            ..Default::default()
        })
        .await
        .unwrap();

        let payload = b"kizzasi-io udp loopback";
        sender.send(payload).await.unwrap();

        let (data, from) = tokio::time::timeout(Duration::from_secs(5), receiver.recv())
            .await
            .expect("UDP receive timed out")
            .unwrap();
        assert_eq!(data.as_ref(), payload);
        // The sender binds to the wildcard address, so only the port is
        // comparable against its own `local_addr`.
        assert_eq!(from.port(), sender.local_addr().unwrap().port());

        // `send_to` targets an explicit address, so it needs an *unconnected*
        // socket: a socket created with `connect()` rejects it (EISCONN).
        let unconnected = UdpSocketStream::bind(SocketConfig {
            address: "127.0.0.1:0".to_string(),
            ..Default::default()
        })
        .await
        .unwrap();
        unconnected.send_to(b"second", &address).await.unwrap();

        let (data, from) = tokio::time::timeout(Duration::from_secs(5), receiver.recv())
            .await
            .expect("UDP receive timed out")
            .unwrap();
        assert_eq!(data.as_ref(), b"second");
        assert_eq!(from, unconnected.local_addr().unwrap());
    }
}
