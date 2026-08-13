//! ZeroMQ stream support
//!
//! Provides ZeroMQ messaging patterns for distributed streaming, on top of the
//! pure-Rust [`zeromq`](https://crates.io/crates/zeromq) ZMTP implementation
//! (no `libzmq` C dependency).
//!
//! ## Features
//! - SUB/PUB pattern support, with server-side topic filtering
//! - PUSH/PULL pattern support
//! - REQ/REP and DEALER pattern support
//! - One socket per stream, created at [`ZmqStream::connect`] and reused for
//!   every send/receive
//! - Send/receive deadlines and a bounded linger on close
//!
//! ## Socket lifecycle
//!
//! The socket is created once, when the stream is connected, and lives for as
//! long as the [`ZmqStream`] does. A previous version created a brand new
//! socket inside every `recv()`/`send()` call -- a fresh TCP connection and
//! ZMTP handshake per message, with anything already queued discarded when the
//! socket was dropped at the end of the call, and a PUB endpoint re-bound on
//! every publish.
//!
//! By convention `Pub` and `Rep` bind the endpoint and every other pattern
//! connects to it; set [`ZmqConfig::bind`] to override.
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
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::info;
use zeromq::{
    DealerSocket, PubSocket, PullSocket, PushSocket, RepSocket, ReqSocket, Socket, SocketRecv,
    SocketSend, SubSocket, ZmqError, ZmqMessage as RawMessage, ZmqResult,
};

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
    /// Request (sends to REP, then receives the reply)
    Req,
    /// Reply (receives from REQ, then sends the reply)
    Rep,
    /// Dealer (asynchronous REQ: send and receive in any order)
    Dealer,
}

impl ZmqPattern {
    /// Whether this pattern binds the endpoint by default (the others connect).
    fn binds_by_default(self) -> bool {
        matches!(self, ZmqPattern::Pub | ZmqPattern::Rep)
    }

    /// Whether the pattern can receive at all (a ZMTP property of the socket
    /// type, not a gap in this implementation).
    fn can_receive(self) -> bool {
        matches!(
            self,
            ZmqPattern::Sub
                | ZmqPattern::Pull
                | ZmqPattern::Req
                | ZmqPattern::Rep
                | ZmqPattern::Dealer
        )
    }

    /// Whether the pattern can send at all.
    fn can_send(self) -> bool {
        matches!(
            self,
            ZmqPattern::Pub
                | ZmqPattern::Push
                | ZmqPattern::Req
                | ZmqPattern::Rep
                | ZmqPattern::Dealer
        )
    }
}

/// ZeroMQ configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZmqConfig {
    /// ZeroMQ endpoint (e.g., "tcp://localhost:5555")
    pub endpoint: String,

    /// Messaging pattern
    pub pattern: ZmqPattern,

    /// Topics to subscribe to (for SUB pattern).
    ///
    /// Each entry becomes a ZMTP subscription prefix. An empty list
    /// subscribes to everything.
    #[serde(default)]
    pub topics: Vec<String>,

    /// Bind (`Some(true)`) or connect (`Some(false)`) the endpoint.
    ///
    /// `None` uses the pattern's conventional role: `Pub` and `Rep` bind,
    /// everything else connects.
    #[serde(default)]
    pub bind: Option<bool>,

    /// Receive deadline (ms) - 0 for no deadline
    #[serde(default)]
    pub recv_timeout_ms: u64,

    /// Send deadline (ms) - 0 for no deadline
    #[serde(default)]
    pub send_timeout_ms: u64,

    /// How long [`ZmqStream::close`] waits for the socket to shut down
    /// cleanly, in milliseconds.
    #[serde(default = "default_linger")]
    pub linger_ms: u64,
}

fn default_linger() -> u64 {
    1000
}

impl Default for ZmqConfig {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            pattern: ZmqPattern::Sub,
            topics: Vec::new(),
            bind: None,
            recv_timeout_ms: 0,
            send_timeout_ms: 0,
            linger_ms: default_linger(),
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

    /// Convert to a wire-level multipart message
    fn to_raw(&self) -> IoResult<RawMessage> {
        let mut parts: Vec<Bytes> = Vec::with_capacity(self.frames.len() + 2);

        // Add topic if present
        if let Some(ref topic) = self.topic {
            parts.push(Bytes::copy_from_slice(topic.as_bytes()));
        }

        // Add payload
        parts.push(self.payload.clone());

        // Add additional frames
        parts.extend(self.frames.iter().cloned());

        RawMessage::try_from(parts)
            .map_err(|_| IoError::Protocol("ZeroMQ message has no frames".to_string()))
    }

    /// Create from a wire-level multipart message
    fn from_raw(raw: RawMessage, has_topic: bool) -> IoResult<Self> {
        let mut parts: Vec<Bytes> = raw.into_vec();

        if parts.is_empty() {
            return Err(IoError::Protocol("Empty ZeroMQ message".to_string()));
        }

        let topic = if has_topic {
            let topic_bytes = parts.remove(0);
            Some(
                String::from_utf8(topic_bytes.to_vec())
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

        let payload = parts.remove(0);
        let frames = parts;

        Ok(Self {
            topic,
            payload,
            frames,
        })
    }
}

/// One live socket, one variant per supported pattern.
enum ZmqSocket {
    Sub(SubSocket),
    Pub(PubSocket),
    Pull(PullSocket),
    Push(PushSocket),
    Req(ReqSocket),
    Rep(RepSocket),
    Dealer(DealerSocket),
}

/// Run `bind`/`connect`/`send`/`recv` over every socket variant.
macro_rules! on_socket {
    ($socket:expr, $name:ident => $body:expr) => {
        match $socket {
            ZmqSocket::Sub($name) => $body,
            ZmqSocket::Pub($name) => $body,
            ZmqSocket::Pull($name) => $body,
            ZmqSocket::Push($name) => $body,
            ZmqSocket::Req($name) => $body,
            ZmqSocket::Rep($name) => $body,
            ZmqSocket::Dealer($name) => $body,
        }
    };
}

impl ZmqSocket {
    fn new(pattern: ZmqPattern) -> Self {
        match pattern {
            ZmqPattern::Sub => ZmqSocket::Sub(SubSocket::new()),
            ZmqPattern::Pub => ZmqSocket::Pub(PubSocket::new()),
            ZmqPattern::Pull => ZmqSocket::Pull(PullSocket::new()),
            ZmqPattern::Push => ZmqSocket::Push(PushSocket::new()),
            ZmqPattern::Req => ZmqSocket::Req(ReqSocket::new()),
            ZmqPattern::Rep => ZmqSocket::Rep(RepSocket::new()),
            ZmqPattern::Dealer => ZmqSocket::Dealer(DealerSocket::new()),
        }
    }

    /// Bind or connect, returning the resolved endpoint when binding (which
    /// makes `tcp://127.0.0.1:0` usable: the OS-assigned port is reported
    /// back).
    async fn attach(&mut self, endpoint: &str, bind: bool) -> ZmqResult<String> {
        if bind {
            on_socket!(self, socket => socket.bind(endpoint).await.map(|e| e.to_string()))
        } else {
            on_socket!(self, socket => socket.connect(endpoint).await.map(|()| endpoint.to_string()))
        }
    }

    async fn close(self) -> Vec<ZmqError> {
        on_socket!(self, socket => socket.close().await)
    }
}

/// ZeroMQ stream: one long-lived socket plus its configuration.
pub struct ZmqStream {
    config: ZmqConfig,
    socket: ZmqSocket,
    endpoint: String,
}

impl ZmqStream {
    /// Connect (or bind) a ZeroMQ endpoint and create the socket.
    ///
    /// Unlike the previous implementation, this really creates a socket and
    /// attaches it to the endpoint, so an unusable endpoint fails here instead
    /// of silently "succeeding".
    pub async fn connect(config: ZmqConfig) -> IoResult<Self> {
        if config.endpoint.trim().is_empty() {
            return Err(IoError::ConfigError(
                "ZeroMQ endpoint must not be empty".to_string(),
            ));
        }

        let bind = config
            .bind
            .unwrap_or_else(|| config.pattern.binds_by_default());
        info!(
            "{} ZeroMQ endpoint: {} (pattern: {:?})",
            if bind { "Binding" } else { "Connecting to" },
            config.endpoint,
            config.pattern
        );

        let mut socket = ZmqSocket::new(config.pattern);
        let endpoint = socket.attach(&config.endpoint, bind).await.map_err(|e| {
            IoError::Connection(format!(
                "Failed to {} ZeroMQ endpoint {}: {}",
                if bind { "bind" } else { "connect" },
                config.endpoint,
                e
            ))
        })?;

        // Apply the SUB topic filter. Without any explicit topic a SUB socket
        // receives nothing at all in ZMTP, so an empty list means "everything".
        if let ZmqSocket::Sub(ref mut sub) = socket {
            if config.topics.is_empty() {
                sub.subscribe("")
                    .await
                    .map_err(|e| IoError::Connection(format!("SUB subscribe failed: {}", e)))?;
            } else {
                for topic in &config.topics {
                    sub.subscribe(topic).await.map_err(|e| {
                        IoError::Connection(format!("SUB subscribe to '{}' failed: {}", topic, e))
                    })?;
                }
            }
        }

        Ok(Self {
            config,
            socket,
            endpoint,
        })
    }

    /// The endpoint the socket is actually attached to.
    ///
    /// When binding to port 0 this is the concrete endpoint chosen by the OS.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Receive a message from the stream.
    ///
    /// Returns [`IoError::Unsupported`] for PUB/PUSH, which cannot receive in
    /// ZMTP at all.
    pub async fn recv(&mut self) -> IoResult<Option<ZmqMessage>> {
        let pattern = self.config.pattern;
        if !pattern.can_receive() {
            return Err(IoError::Unsupported(format!(
                "A {:?} socket cannot receive (ZeroMQ pattern is send-only)",
                pattern
            )));
        }

        let timeout_ms = self.config.recv_timeout_ms;
        let raw = match self.socket {
            ZmqSocket::Sub(ref mut socket) => with_deadline(timeout_ms, socket.recv()).await,
            ZmqSocket::Pull(ref mut socket) => with_deadline(timeout_ms, socket.recv()).await,
            ZmqSocket::Req(ref mut socket) => with_deadline(timeout_ms, socket.recv()).await,
            ZmqSocket::Rep(ref mut socket) => with_deadline(timeout_ms, socket.recv()).await,
            ZmqSocket::Dealer(ref mut socket) => with_deadline(timeout_ms, socket.recv()).await,
            ZmqSocket::Pub(_) | ZmqSocket::Push(_) => {
                return Err(IoError::Unsupported(format!(
                    "A {:?} socket cannot receive (ZeroMQ pattern is send-only)",
                    pattern
                )))
            }
        }
        .map_err(|e| map_recv_error(e, timeout_ms))?;

        // Only SUB messages carry a leading topic frame.
        let msg = ZmqMessage::from_raw(raw, pattern == ZmqPattern::Sub)?;
        Ok(Some(msg))
    }

    /// Send a message on the stream.
    ///
    /// Returns [`IoError::Unsupported`] for SUB/PULL, which cannot send in
    /// ZMTP at all.
    pub async fn send(&mut self, msg: ZmqMessage) -> IoResult<()> {
        let pattern = self.config.pattern;
        if !pattern.can_send() {
            return Err(IoError::Unsupported(format!(
                "A {:?} socket cannot send (ZeroMQ pattern is receive-only)",
                pattern
            )));
        }

        let raw = msg.to_raw()?;
        let timeout_ms = self.config.send_timeout_ms;
        match self.socket {
            ZmqSocket::Pub(ref mut socket) => with_deadline(timeout_ms, socket.send(raw)).await,
            ZmqSocket::Push(ref mut socket) => with_deadline(timeout_ms, socket.send(raw)).await,
            ZmqSocket::Req(ref mut socket) => with_deadline(timeout_ms, socket.send(raw)).await,
            ZmqSocket::Rep(ref mut socket) => with_deadline(timeout_ms, socket.send(raw)).await,
            ZmqSocket::Dealer(ref mut socket) => with_deadline(timeout_ms, socket.send(raw)).await,
            ZmqSocket::Sub(_) | ZmqSocket::Pull(_) => {
                return Err(IoError::Unsupported(format!(
                    "A {:?} socket cannot send (ZeroMQ pattern is receive-only)",
                    pattern
                )))
            }
        }
        .map_err(|e| map_send_error(e, timeout_ms))
    }

    /// Add a topic subscription to a live SUB socket.
    pub async fn subscribe(&mut self, topic: &str) -> IoResult<()> {
        match self.socket {
            ZmqSocket::Sub(ref mut socket) => socket
                .subscribe(topic)
                .await
                .map_err(|e| IoError::Connection(format!("SUB subscribe failed: {}", e))),
            _ => Err(IoError::Unsupported(
                "Only a SUB socket can subscribe to topics".to_string(),
            )),
        }
    }

    /// Remove a topic subscription from a live SUB socket.
    pub async fn unsubscribe(&mut self, topic: &str) -> IoResult<()> {
        match self.socket {
            ZmqSocket::Sub(ref mut socket) => socket
                .unsubscribe(topic)
                .await
                .map_err(|e| IoError::Connection(format!("SUB unsubscribe failed: {}", e))),
            _ => Err(IoError::Unsupported(
                "Only a SUB socket can unsubscribe from topics".to_string(),
            )),
        }
    }

    /// Close the socket, waiting at most `linger_ms` for a clean shutdown.
    pub async fn close(self) -> IoResult<()> {
        let linger = Duration::from_millis(self.config.linger_ms);
        let errors = if self.config.linger_ms == 0 {
            self.socket.close().await
        } else {
            match tokio::time::timeout(linger, self.socket.close()).await {
                Ok(errors) => errors,
                Err(_) => {
                    return Err(IoError::StreamError(format!(
                        "ZeroMQ socket did not close within the {} ms linger period",
                        self.config.linger_ms
                    )))
                }
            }
        };

        match errors.into_iter().next() {
            Some(e) => Err(IoError::Connection(format!(
                "ZeroMQ socket close failed: {}",
                e
            ))),
            None => Ok(()),
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &ZmqConfig {
        &self.config
    }
}

/// Apply an optional deadline to a socket operation. `0` means "no deadline".
async fn with_deadline<T, F>(timeout_ms: u64, future: F) -> Result<T, DeadlineError>
where
    F: std::future::Future<Output = ZmqResult<T>>,
{
    if timeout_ms == 0 {
        future.await.map_err(DeadlineError::Zmq)
    } else {
        match tokio::time::timeout(Duration::from_millis(timeout_ms), future).await {
            Ok(result) => result.map_err(DeadlineError::Zmq),
            Err(_) => Err(DeadlineError::Elapsed),
        }
    }
}

/// Either the socket failed, or the configured deadline elapsed first.
enum DeadlineError {
    Zmq(ZmqError),
    Elapsed,
}

fn map_recv_error(error: DeadlineError, timeout_ms: u64) -> IoError {
    match error {
        DeadlineError::Zmq(e) => IoError::Connection(format!("ZeroMQ receive failed: {}", e)),
        DeadlineError::Elapsed => {
            IoError::StreamError(format!("ZeroMQ receive timed out after {} ms", timeout_ms))
        }
    }
}

fn map_send_error(error: DeadlineError, timeout_ms: u64) -> IoError {
    match error {
        DeadlineError::Zmq(e) => IoError::Connection(format!("ZeroMQ send failed: {}", e)),
        DeadlineError::Elapsed => {
            IoError::StreamError(format!("ZeroMQ send timed out after {} ms", timeout_ms))
        }
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
        assert_eq!(config.linger_ms, 1000);
        assert_eq!(config.recv_timeout_ms, 0);
        assert!(config.bind.is_none());
        // `high_water_mark`, `reconnect` and `reconnect_interval_ms` used to
        // live here but never reached a socket. They were removed rather than
        // left as accepted-and-ignored keys; the pure-Rust ZMTP backend
        // manages its own queueing and reconnects SUB peers automatically.
    }

    #[test]
    fn test_zmq_message_round_trips_through_the_wire_format() {
        let mut msg = ZmqMessage::with_topic("sensor", &b"payload"[..]);
        msg.add_frame(&b"extra"[..]);

        let raw = msg.to_raw().expect("non-empty message");
        let back = ZmqMessage::from_raw(raw, true).expect("decodes");

        assert_eq!(back.topic, Some("sensor".to_string()));
        assert_eq!(back.payload, Bytes::from(&b"payload"[..]));
        assert_eq!(back.frames, vec![Bytes::from(&b"extra"[..])]);
    }

    // === Regression tests: real sockets, real patterns (id=28/303/273/357) ===

    #[tokio::test]
    async fn test_connect_rejects_an_empty_endpoint() {
        let config = ZmqConfig {
            endpoint: String::new(),
            ..Default::default()
        };
        assert!(ZmqStream::connect(config).await.is_err());
    }

    #[tokio::test]
    async fn test_connect_fails_on_an_unusable_endpoint() {
        // A previous version only constructed a tmq Context and returned Ok,
        // so *any* endpoint string "connected" successfully.
        let config = ZmqConfig {
            endpoint: "definitely-not-a-zmq-endpoint".to_string(),
            pattern: ZmqPattern::Pub,
            bind: Some(true),
            ..Default::default()
        };
        assert!(ZmqStream::connect(config).await.is_err());
    }

    #[tokio::test]
    async fn test_push_pull_round_trip_over_a_persistent_socket() {
        let mut puller = ZmqStream::connect(ZmqConfig {
            endpoint: "tcp://127.0.0.1:0".to_string(),
            pattern: ZmqPattern::Pull,
            bind: Some(true),
            recv_timeout_ms: 5_000,
            ..Default::default()
        })
        .await
        .expect("bind PULL");

        let endpoint = puller.endpoint().to_string();
        let mut pusher = ZmqStream::connect(ZmqConfig {
            endpoint: endpoint.clone(),
            pattern: ZmqPattern::Push,
            send_timeout_ms: 5_000,
            ..Default::default()
        })
        .await
        .expect("connect PUSH");

        for i in 0..3u8 {
            pusher
                .send(ZmqMessage::new(vec![i, i + 1]))
                .await
                .expect("push send");
        }

        // Every message must arrive: the old per-call socket was dropped
        // right after one receive, discarding whatever was still queued.
        for i in 0..3u8 {
            let msg = puller.recv().await.expect("pull recv").expect("a message");
            assert_eq!(msg.payload, Bytes::from(vec![i, i + 1]));
            assert!(msg.topic.is_none());
        }

        pusher.close().await.expect("close PUSH");
        puller.close().await.expect("close PULL");
    }

    #[tokio::test]
    async fn test_pub_sub_delivers_with_topic_filtering() {
        let mut publisher = ZmqStream::connect(ZmqConfig {
            endpoint: "tcp://127.0.0.1:0".to_string(),
            pattern: ZmqPattern::Pub,
            send_timeout_ms: 5_000,
            ..Default::default()
        })
        .await
        .expect("bind PUB");

        let endpoint = publisher.endpoint().to_string();
        let mut subscriber = ZmqStream::connect(ZmqConfig {
            endpoint,
            pattern: ZmqPattern::Sub,
            topics: vec!["sensor".to_string()],
            recv_timeout_ms: 5_000,
            ..Default::default()
        })
        .await
        .expect("connect SUB");

        // PUB/SUB is a slow joiner: republish until the subscription has
        // propagated. `recv_sub` used to return Unsupported unconditionally,
        // so the crate's own documented example could never work.
        let received = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                publisher
                    .send(ZmqMessage::with_topic("other", &b"ignored"[..]))
                    .await
                    .expect("publish filtered-out topic");
                publisher
                    .send(ZmqMessage::with_topic("sensor", &b"42"[..]))
                    .await
                    .expect("publish");

                if let Ok(Ok(Some(msg))) =
                    tokio::time::timeout(Duration::from_millis(200), subscriber.recv()).await
                {
                    return msg;
                }
            }
        })
        .await
        .expect("SUB should receive the published message");

        assert_eq!(received.topic, Some("sensor".to_string()));
        assert_eq!(received.payload, Bytes::from(&b"42"[..]));

        subscriber.close().await.expect("close SUB");
        publisher.close().await.expect("close PUB");
    }

    #[tokio::test]
    async fn test_req_rep_round_trip() {
        // REQ/REP used to return Unsupported("not fully implemented yet")
        // from all four of recv_rep/recv_dealer/send_req/send_rep.
        let mut replier = ZmqStream::connect(ZmqConfig {
            endpoint: "tcp://127.0.0.1:0".to_string(),
            pattern: ZmqPattern::Rep,
            recv_timeout_ms: 5_000,
            send_timeout_ms: 5_000,
            ..Default::default()
        })
        .await
        .expect("bind REP");

        let endpoint = replier.endpoint().to_string();
        let requester = tokio::spawn(async move {
            let mut req = ZmqStream::connect(ZmqConfig {
                endpoint,
                pattern: ZmqPattern::Req,
                recv_timeout_ms: 5_000,
                send_timeout_ms: 5_000,
                ..Default::default()
            })
            .await
            .expect("connect REQ");

            req.send(ZmqMessage::new(&b"ping"[..]))
                .await
                .expect("REQ send");
            let reply = req.recv().await.expect("REQ recv").expect("a reply");
            req.close().await.expect("close REQ");
            reply.payload
        });

        let request = replier.recv().await.expect("REP recv").expect("a request");
        assert_eq!(request.payload, Bytes::from(&b"ping"[..]));
        replier
            .send(ZmqMessage::new(&b"pong"[..]))
            .await
            .expect("REP send");

        let reply = requester.await.expect("requester task");
        assert_eq!(reply, Bytes::from(&b"pong"[..]));

        replier.close().await.expect("close REP");
    }

    #[tokio::test]
    async fn test_send_only_and_receive_only_patterns_report_unsupported() {
        let mut publisher = ZmqStream::connect(ZmqConfig {
            endpoint: "tcp://127.0.0.1:0".to_string(),
            pattern: ZmqPattern::Pub,
            ..Default::default()
        })
        .await
        .expect("bind PUB");
        assert!(matches!(
            publisher.recv().await,
            Err(IoError::Unsupported(_))
        ));

        let endpoint = publisher.endpoint().to_string();
        let mut subscriber = ZmqStream::connect(ZmqConfig {
            endpoint,
            pattern: ZmqPattern::Sub,
            ..Default::default()
        })
        .await
        .expect("connect SUB");
        assert!(matches!(
            subscriber.send(ZmqMessage::new(&b"x"[..])).await,
            Err(IoError::Unsupported(_))
        ));

        subscriber.close().await.expect("close SUB");
        publisher.close().await.expect("close PUB");
    }

    #[tokio::test]
    async fn test_recv_timeout_is_actually_applied() {
        // `recv_timeout_ms` was declared and defaulted but never reached a
        // socket, so a stalled peer hung the caller forever.
        let mut puller = ZmqStream::connect(ZmqConfig {
            endpoint: "tcp://127.0.0.1:0".to_string(),
            pattern: ZmqPattern::Pull,
            bind: Some(true),
            recv_timeout_ms: 150,
            ..Default::default()
        })
        .await
        .expect("bind PULL");

        let started = std::time::Instant::now();
        let result = puller.recv().await;
        assert!(
            matches!(result, Err(IoError::StreamError(_))),
            "expected a timeout error, got {result:?}"
        );
        assert!(started.elapsed() < Duration::from_secs(5));

        puller.close().await.expect("close PULL");
    }
}
