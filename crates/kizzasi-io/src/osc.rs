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
use rosc::{
    OscArray, OscBundle, OscColor, OscMessage as RoscMessage, OscMidiMessage, OscPacket, OscTime,
    OscType,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    /// MIDI message: (port id, status byte, data1, data2)
    Midi {
        port: u8,
        status: u8,
        data1: u8,
        data2: u8,
    },
    /// OSC time tag: NTP-style (seconds, fractional-seconds) since 1900-01-01
    Time { seconds: u32, fractional: u32 },
    /// RGBA color
    Color {
        red: u8,
        green: u8,
        blue: u8,
        alpha: u8,
    },
    /// Nested array of arguments
    Array(Vec<OscArg>),
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
            // rosc's OscType::Inf is OSC's 'I' (impulse/bang) tag; encoding
            // maps OscArg::Impulse back to OscType::Inf below, so this must
            // be the symmetric inverse. A previous version mapped Inf to
            // OscArg::Double(f64::INFINITY), which never round-tripped and
            // made OscArg::Impulse unreachable from any received message.
            OscType::Inf => OscArg::Impulse,
            OscType::Time(t) => OscArg::Time {
                seconds: t.seconds,
                fractional: t.fractional,
            },
            OscType::Char(c) => OscArg::String(c.to_string()),
            OscType::Color(c) => OscArg::Color {
                red: c.red,
                green: c.green,
                blue: c.blue,
                alpha: c.alpha,
            },
            OscType::Midi(m) => OscArg::Midi {
                port: m.port,
                status: m.status,
                data1: m.data1,
                data2: m.data2,
            },
            OscType::Array(arr) => {
                OscArg::Array(arr.content.into_iter().map(OscArg::from).collect())
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
            OscArg::Midi {
                port,
                status,
                data1,
                data2,
            } => OscType::Midi(OscMidiMessage {
                port,
                status,
                data1,
                data2,
            }),
            OscArg::Time {
                seconds,
                fractional,
            } => OscType::Time(OscTime {
                seconds,
                fractional,
            }),
            OscArg::Color {
                red,
                green,
                blue,
                alpha,
            } => OscType::Color(OscColor {
                red,
                green,
                blue,
                alpha,
            }),
            OscArg::Array(items) => OscType::Array(OscArray {
                content: items.into_iter().map(OscType::from).collect(),
            }),
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
    /// Messages already decoded from a previous packet (typically a
    /// bundle) but not yet returned by `recv()`. A previous version
    /// returned only `bundle.content`'s first message and silently
    /// dropped the rest, and returned `Ok(None)` outright whenever that
    /// first element was itself a nested bundle. `recv()` now flattens any
    /// (possibly nested) bundle into this queue up front and drains it one
    /// message per call, so no message is lost and nested bundles are
    /// handled recursively.
    pending: VecDeque<OscMessage>,
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
            pending: VecDeque::new(),
        })
    }

    /// Receive an OSC message. Bundles are flattened (recursively, in case
    /// a bundle contains nested bundles) into individual messages that are
    /// returned one at a time across successive calls; a genuinely empty
    /// bundle simply yields no message on this call without erroring.
    pub async fn recv(&mut self) -> IoResult<Option<OscMessage>> {
        if let Some(msg) = self.pending.pop_front() {
            return Ok(Some(msg));
        }

        let (size, _addr) =
            self.socket.recv_from(&mut self.buffer).await.map_err(|e| {
                IoError::ReadFailed(format!("Failed to receive OSC message: {}", e))
            })?;

        let packet = rosc::decoder::decode_udp(&self.buffer[..size])
            .map_err(|e| IoError::ReadFailed(format!("Failed to decode OSC packet: {}", e)))?;

        Self::flatten_packet(packet.1, &mut self.pending);

        if let Some(msg) = self.pending.front() {
            debug!("Received OSC message: {}", msg.address);
        }

        Ok(self.pending.pop_front())
    }

    /// Recursively flatten an OSC packet (a single message, or a bundle
    /// that may itself contain nested bundles) into individual messages,
    /// appended to `out` in order.
    fn flatten_packet(packet: OscPacket, out: &mut VecDeque<OscMessage>) {
        match packet {
            OscPacket::Message(msg) => {
                out.push_back(OscMessage {
                    address: msg.addr,
                    args: msg.args.into_iter().map(OscArg::from).collect(),
                });
            }
            OscPacket::Bundle(bundle) => {
                for inner in bundle.content {
                    Self::flatten_packet(inner, out);
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

    /// Number of already-decoded messages waiting to be returned by
    /// `recv()` (e.g. the remainder of a previously received bundle).
    pub fn pending_len(&self) -> usize {
        self.pending.len()
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
    ///
    /// Runs until an unrecoverable I/O error occurs (e.g. the socket
    /// itself fails). A single malformed/undecodable UDP datagram no
    /// longer terminates the loop: a previous version propagated
    /// `recv()`'s error with `?`, so any peer on the network sending one
    /// bad packet would permanently kill the server.
    pub async fn run(&mut self) -> IoResult<()> {
        info!("OSC server started");

        loop {
            match self.receiver.recv().await {
                Ok(Some(msg)) => {
                    // Find matching handlers
                    let mut handled = false;
                    for (pattern, handler) in &self.handlers {
                        if Self::matches_pattern(&msg.address, pattern) {
                            handler(&msg);
                            handled = true;
                        }
                    }

                    if !handled {
                        warn!("Unhandled OSC message: {}", msg.address);
                    }
                }
                Ok(None) => continue,
                Err(e) => {
                    warn!(
                        "OSC server: failed to receive/decode a packet, continuing: {}",
                        e
                    );
                    continue;
                }
            }
        }
    }

    /// OSC 1.0 address pattern matching.
    ///
    /// Supports `?` (any single character), `*` (any run of zero or more
    /// characters), `[...]` character classes (`[abc]`, `[!abc]` negation,
    /// `[a-z]` ranges), and `{foo,bar}` alternation, per the OSC address
    /// pattern spec. A previous version only handled a single `*` via
    /// `starts_with`/`ends_with`, which both ignored every other operator
    /// and matched incorrectly when the prefix/suffix could overlap (e.g.
    /// pattern `/aaa*aaa` against address `/aaa`: `starts_with("/aaa")` and
    /// `ends_with("aaa")` are both trivially true for the SAME four
    /// characters, so it reported a match despite there being no `*`-worth
    /// of characters between them).
    fn matches_pattern(address: &str, pattern: &str) -> bool {
        match_osc_pattern(address.as_bytes(), pattern.as_bytes())
    }
}

/// Recursive backtracking matcher behind `OscServer::matches_pattern`.
/// Operates on bytes (OSC addresses/patterns are ASCII) so no UTF-8
/// char-boundary slicing is needed. Every slice index used below is
/// established as in-bounds either by a preceding `first()`/`position()`
/// check or by short-circuit evaluation of `!slice.is_empty() && ...`
/// immediately before the index.
fn match_osc_pattern(addr: &[u8], pat: &[u8]) -> bool {
    match pat.first() {
        None => addr.is_empty(),
        Some(b'*') => {
            // Zero-or-more: try matching the remaining pattern at every
            // possible split point in `addr`.
            (0..=addr.len()).any(|i| match_osc_pattern(&addr[i..], &pat[1..]))
        }
        Some(b'?') => !addr.is_empty() && match_osc_pattern(&addr[1..], &pat[1..]),
        Some(b'[') => match_osc_class(addr, pat),
        Some(b'{') => match_osc_alternation(addr, pat),
        Some(&literal) => {
            !addr.is_empty() && addr[0] == literal && match_osc_pattern(&addr[1..], &pat[1..])
        }
    }
}

/// Handle a `[...]`/`[!...]` character class at the start of `pat`
/// (`pat[0] == b'['`), matching it against `addr[0]`.
fn match_osc_class(addr: &[u8], pat: &[u8]) -> bool {
    let Some(close) = pat.iter().position(|&b| b == b']') else {
        // Unterminated class: treat '[' as a literal character.
        return !addr.is_empty() && addr[0] == b'[' && match_osc_pattern(&addr[1..], &pat[1..]);
    };
    if addr.is_empty() {
        return false;
    }
    let class = &pat[1..close];
    let (negate, class) = match class.first() {
        Some(b'!') => (true, &class[1..]),
        _ => (false, class),
    };
    if class_contains(class, addr[0]) == negate {
        return false;
    }
    match_osc_pattern(&addr[1..], &pat[close + 1..])
}

/// Whether `class` (the contents of a `[...]` block, with any leading `!`
/// already stripped) contains byte `c`, honoring `a-z`-style ranges.
fn class_contains(class: &[u8], c: u8) -> bool {
    let mut i = 0;
    while i < class.len() {
        if i + 2 < class.len() && class[i + 1] == b'-' {
            let (lo, hi) = (class[i], class[i + 2]);
            if lo <= c && c <= hi {
                return true;
            }
            i += 3;
        } else {
            if class[i] == c {
                return true;
            }
            i += 1;
        }
    }
    false
}

/// Handle a `{foo,bar,...}` alternation at the start of `pat`
/// (`pat[0] == b'{'`).
fn match_osc_alternation(addr: &[u8], pat: &[u8]) -> bool {
    let Some(close) = pat.iter().position(|&b| b == b'}') else {
        // Unterminated alternation: treat '{' as a literal character.
        return !addr.is_empty() && addr[0] == b'{' && match_osc_pattern(&addr[1..], &pat[1..]);
    };
    let rest = &pat[close + 1..];
    pat[1..close].split(|&b| b == b',').any(|alt| {
        addr.len() >= alt.len()
            && &addr[..alt.len()] == alt
            && match_osc_pattern(&addr[alt.len()..], rest)
    })
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

    // === Regression tests: MIDI/Color/Time/Array + Impulse round trip (medium, id=46) ===

    #[test]
    fn test_impulse_round_trips_symmetrically() {
        // A previous version decoded OscType::Inf to
        // OscArg::Double(f64::INFINITY) while encoding OscArg::Impulse to
        // OscType::Inf, so Impulse could never round-trip and was
        // unreachable from any received message.
        let arg = OscArg::Impulse;
        let osc_type: OscType = arg.clone().into();
        assert!(matches!(osc_type, OscType::Inf));
        let back: OscArg = osc_type.into();
        assert_eq!(back, arg);
    }

    #[test]
    fn test_midi_round_trips_without_data_loss() {
        let arg = OscArg::Midi {
            port: 1,
            status: 0x90,
            data1: 60,
            data2: 127,
        };
        let osc_type: OscType = arg.clone().into();
        let back: OscArg = osc_type.into();
        assert_eq!(
            back, arg,
            "MIDI bytes must survive the round trip instead of being discarded to an empty Blob"
        );
    }

    #[test]
    fn test_color_round_trips_without_data_loss() {
        let arg = OscArg::Color {
            red: 10,
            green: 20,
            blue: 30,
            alpha: 255,
        };
        let osc_type: OscType = arg.clone().into();
        let back: OscArg = osc_type.into();
        assert_eq!(back, arg);
    }

    #[test]
    fn test_time_round_trips_without_data_loss() {
        let arg = OscArg::Time {
            seconds: 123_456,
            fractional: 789,
        };
        let osc_type: OscType = arg.clone().into();
        let back: OscArg = osc_type.into();
        assert_eq!(
            back, arg,
            "time tag must survive the round trip instead of collapsing to a constant 0"
        );
    }

    #[test]
    fn test_array_round_trips_all_elements_not_just_the_first() {
        let arg = OscArg::Array(vec![
            OscArg::Int(1),
            OscArg::Float(2.5),
            OscArg::String("three".to_string()),
        ]);
        let osc_type: OscType = arg.clone().into();
        let back: OscArg = osc_type.into();
        assert_eq!(
            back, arg,
            "every array element must survive, not just the first"
        );
    }

    // === Regression tests: bundle flattening (medium, id=46) ===

    #[tokio::test]
    async fn test_recv_returns_every_message_in_a_bundle() {
        let mut receiver = OscReceiver::new("127.0.0.1:0").await.unwrap();
        let actual_addr = receiver.local_addr().unwrap();
        let sender = OscSender::new(&actual_addr.to_string()).await.unwrap();

        let messages = vec![
            OscMessage {
                address: "/a".to_string(),
                args: vec![OscArg::Int(1)],
            },
            OscMessage {
                address: "/b".to_string(),
                args: vec![OscArg::Int(2)],
            },
            OscMessage {
                address: "/c".to_string(),
                args: vec![OscArg::Int(3)],
            },
        ];
        sender.send_bundle(messages).await.unwrap();

        // A previous version returned only the bundle's first message and
        // silently dropped the rest.
        let mut received_addrs = Vec::new();
        for _ in 0..3 {
            let msg = receiver
                .recv_timeout(std::time::Duration::from_secs(2))
                .await
                .unwrap()
                .expect("expected a message from the bundle");
            received_addrs.push(msg.address);
        }
        assert_eq!(received_addrs, vec!["/a", "/b", "/c"]);
    }

    #[test]
    fn test_flatten_packet_handles_nested_bundles() {
        // A previous version returned Ok(None) outright whenever the
        // bundle's first element was itself a nested bundle.
        let inner_bundle = OscPacket::Bundle(OscBundle {
            timetag: (0, 0).into(),
            content: vec![OscPacket::Message(RoscMessage {
                addr: "/nested".to_string(),
                args: vec![],
            })],
        });
        let outer = OscPacket::Bundle(OscBundle {
            timetag: (0, 0).into(),
            content: vec![
                inner_bundle,
                OscPacket::Message(RoscMessage {
                    addr: "/top".to_string(),
                    args: vec![],
                }),
            ],
        });

        let mut out = VecDeque::new();
        OscReceiver::flatten_packet(outer, &mut out);
        let addrs: Vec<&str> = out.iter().map(|m| m.address.as_str()).collect();
        assert_eq!(addrs, vec!["/nested", "/top"]);
    }

    // === Regression test: server loop survives a malformed packet (medium, id=46) ===

    #[tokio::test]
    async fn test_server_run_survives_malformed_packet() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use tokio::net::UdpSocket as TokioUdpSocket;

        let mut server = OscServer::new("127.0.0.1:0").await.unwrap();
        let server_addr = server.receiver.local_addr().unwrap();

        let hit_count = Arc::new(AtomicUsize::new(0));
        let hit_count_clone = hit_count.clone();
        server.add_handler("/ok", move |_msg| {
            hit_count_clone.fetch_add(1, Ordering::SeqCst);
        });

        let run_handle = tokio::spawn(async move {
            let _ = server.run().await;
        });

        let client = TokioUdpSocket::bind("127.0.0.1:0").await.unwrap();

        // A previous version propagated recv()'s error with `?`, so this
        // single undecodable datagram would have permanently killed the
        // server loop.
        client
            .send_to(b"not a valid OSC packet", server_addr)
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // The server must still be alive and processing subsequent valid
        // messages.
        let good_sender = OscSender::new(&server_addr.to_string()).await.unwrap();
        good_sender.send_float("/ok", 1.0).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        assert_eq!(
            hit_count.load(Ordering::SeqCst),
            1,
            "server should have handled the valid message after surviving the malformed one"
        );

        run_handle.abort();
    }

    // === Regression tests: OSC address pattern matching (medium, id=46) ===

    #[test]
    fn test_pattern_star_overlap_bug_is_fixed() {
        // pattern "/aaa*aaa" against address "/aaa": a naive
        // starts_with("/aaa") && ends_with("aaa") check reports a match by
        // reusing the SAME four characters for both the prefix and the
        // suffix, even though there is no `*`-worth of characters between
        // them. The correct backtracking matcher must reject this.
        assert!(!OscServer::matches_pattern("/aaa", "/aaa*aaa"));
        // But it must still match when there really is a gap.
        assert!(OscServer::matches_pattern("/aaaXaaa", "/aaa*aaa"));
        assert!(OscServer::matches_pattern("/aaaaaa", "/aaa*aaa"));
    }

    #[test]
    fn test_pattern_basic_wildcard_and_literal() {
        assert!(OscServer::matches_pattern("/synth/volume", "*"));
        assert!(OscServer::matches_pattern("/synth/volume", "/synth/*"));
        assert!(!OscServer::matches_pattern("/other/volume", "/synth/*"));
        assert!(OscServer::matches_pattern("/synth/volume", "/synth/volume"));
        assert!(!OscServer::matches_pattern(
            "/synth/volume",
            "/synth/volume2"
        ));
    }

    #[test]
    fn test_pattern_question_mark() {
        assert!(OscServer::matches_pattern("/synth/1", "/synth/?"));
        assert!(!OscServer::matches_pattern("/synth/12", "/synth/?"));
        assert!(!OscServer::matches_pattern("/synth/", "/synth/?"));
    }

    #[test]
    fn test_pattern_character_class() {
        assert!(OscServer::matches_pattern("/synth/1", "/synth/[0-9]"));
        assert!(OscServer::matches_pattern("/synth/a", "/synth/[abc]"));
        assert!(!OscServer::matches_pattern("/synth/d", "/synth/[abc]"));
        assert!(OscServer::matches_pattern("/synth/d", "/synth/[!abc]"));
        assert!(!OscServer::matches_pattern("/synth/a", "/synth/[!abc]"));
    }

    #[test]
    fn test_pattern_alternation() {
        assert!(OscServer::matches_pattern("/synth/foo", "/synth/{foo,bar}"));
        assert!(OscServer::matches_pattern("/synth/bar", "/synth/{foo,bar}"));
        assert!(!OscServer::matches_pattern(
            "/synth/baz",
            "/synth/{foo,bar}"
        ));
    }

    #[test]
    fn test_pattern_empty_and_edge_cases() {
        assert!(OscServer::matches_pattern("", ""));
        assert!(!OscServer::matches_pattern("/a", ""));
        assert!(OscServer::matches_pattern("", "*"));
        assert!(!OscServer::matches_pattern("", "?"));
    }
}
