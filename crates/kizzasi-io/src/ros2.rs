//! ROS2 subscriber bridge for robotic sensor streams
//!
//! Provides integration with ROS2 (Robot Operating System 2) for subscribing to sensor data
//! topics such as IMU, laser scans and numeric arrays.
//!
//! Built on [`ros2-client`](https://crates.io/crates/ros2-client) and RustDDS,
//! both of which are pure Rust: no ROS2 C/C++ installation is required and the
//! feature builds on every platform (the previous `r2r` backend required a
//! full ROS2 install and was unavailable on macOS).
//!
//! ## Features
//!
//! - Subscribe to ROS2 topics with type-safe message handling
//! - Support for common message types (`std_msgs/Float32`, `Float64`,
//!   `Float32MultiArray`, `Float64MultiArray`, `sensor_msgs/Imu`,
//!   `sensor_msgs/LaserScan`)
//! - Async streaming interface compatible with [`AsyncSignalStream`]
//! - Quality of Service (QoS) configuration
//!
//! ## Read semantics
//!
//! [`Ros2Stream`] obeys the crate-wide stream read contract (see the `stream`
//! module): `read()` returns only samples that genuinely arrived on the topic,
//! [`IoError::BufferEmpty`] while a full block has not accumulated yet,
//! [`IoError::EndOfStream`] after `close()`, and [`IoError::NotConnected`]
//! before `start()`. A previous version dropped every subscription the instant
//! it was created and returned `Ok(Array1::zeros(buffer_size))` from every
//! `read()` -- an endless stream of fabricated zeros indistinguishable from a
//! genuinely silent sensor.
//!
//! ## Example
//!
//! ```rust,no_run
//! use kizzasi_io::{AsyncSignalStream, Ros2Config, Ros2MessageType, Ros2Stream};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = Ros2Config::new("/imu/data", Ros2MessageType::Imu);
//!     let mut stream = Ros2Stream::new(config).await?;
//!     stream.start().await?;
//!
//!     while stream.is_active() {
//!         match stream.read().await {
//!             Ok(data) => println!("Received {} samples", data.len()),
//!             Err(e) => {
//!                 eprintln!("no data yet: {e}");
//!                 break;
//!             }
//!         }
//!     }
//!     Ok(())
//! }
//! ```

use crate::error::{IoError, IoResult};
use crate::stream::{AsyncSignalStream, StreamConfig};
use async_trait::async_trait;
use ros2_client::ros2::{policy, Duration as DdsDuration, QosPolicies, QosPolicyBuilder};
use ros2_client::{Context, MessageTypeName, Name, Node, NodeName, NodeOptions, Subscription};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub mod msg {
    //! ROS2 message definitions used by [`Ros2Stream`](super::Ros2Stream).
    //!
    //! Field order and types mirror the official `.msg` interface
    //! definitions (`std_msgs`, `geometry_msgs`, `sensor_msgs`,
    //! `builtin_interfaces`) exactly, because DDS/CDR is a positional
    //! encoding: a reordered or retyped field silently decodes garbage.
    //! Fixed-size arrays (`float64[9]`) are encoded without a length prefix,
    //! which is what `[f64; 9]` produces through serde; variable-length
    //! sequences (`float32[]`) carry a `u32` length, which is what `Vec<_>`
    //! produces.

    use serde::{Deserialize, Serialize};

    /// `builtin_interfaces/Time`
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Time {
        /// Seconds since the epoch
        pub sec: i32,
        /// Nanosecond remainder
        pub nanosec: u32,
    }

    /// `std_msgs/Header`
    #[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Header {
        /// Acquisition timestamp
        pub stamp: Time,
        /// Coordinate frame this data is associated with
        pub frame_id: String,
    }

    /// `geometry_msgs/Vector3`
    #[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
    pub struct Vector3 {
        /// X component
        pub x: f64,
        /// Y component
        pub y: f64,
        /// Z component
        pub z: f64,
    }

    /// `geometry_msgs/Quaternion`
    #[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
    pub struct Quaternion {
        /// X component
        pub x: f64,
        /// Y component
        pub y: f64,
        /// Z component
        pub z: f64,
        /// W component
        pub w: f64,
    }

    /// `std_msgs/MultiArrayDimension`
    #[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct MultiArrayDimension {
        /// Label of this dimension
        pub label: String,
        /// Size of this dimension
        pub size: u32,
        /// Stride of this dimension
        pub stride: u32,
    }

    /// `std_msgs/MultiArrayLayout`
    #[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct MultiArrayLayout {
        /// Array of dimension properties
        pub dim: Vec<MultiArrayDimension>,
        /// Padding elements at the front of the data
        pub data_offset: u32,
    }

    /// `std_msgs/Float32`
    #[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
    pub struct Float32 {
        /// Payload
        pub data: f32,
    }

    /// `std_msgs/Float64`
    #[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
    pub struct Float64 {
        /// Payload
        pub data: f64,
    }

    /// `std_msgs/Float32MultiArray`
    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    pub struct Float32MultiArray {
        /// Layout description
        pub layout: MultiArrayLayout,
        /// Payload
        pub data: Vec<f32>,
    }

    /// `std_msgs/Float64MultiArray`
    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    pub struct Float64MultiArray {
        /// Layout description
        pub layout: MultiArrayLayout,
        /// Payload
        pub data: Vec<f64>,
    }

    /// `sensor_msgs/Imu`
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Imu {
        /// Message header
        pub header: Header,
        /// Orientation estimate
        pub orientation: Quaternion,
        /// Row-major 3x3 orientation covariance
        pub orientation_covariance: [f64; 9],
        /// Angular velocity (rad/s)
        pub angular_velocity: Vector3,
        /// Row-major 3x3 angular velocity covariance
        pub angular_velocity_covariance: [f64; 9],
        /// Linear acceleration (m/s^2)
        pub linear_acceleration: Vector3,
        /// Row-major 3x3 linear acceleration covariance
        pub linear_acceleration_covariance: [f64; 9],
    }

    impl Default for Imu {
        fn default() -> Self {
            Self {
                header: Header::default(),
                orientation: Quaternion::default(),
                orientation_covariance: [0.0; 9],
                angular_velocity: Vector3::default(),
                angular_velocity_covariance: [0.0; 9],
                linear_acceleration: Vector3::default(),
                linear_acceleration_covariance: [0.0; 9],
            }
        }
    }

    /// `sensor_msgs/LaserScan`
    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    pub struct LaserScan {
        /// Message header
        pub header: Header,
        /// Start angle of the scan (rad)
        pub angle_min: f32,
        /// End angle of the scan (rad)
        pub angle_max: f32,
        /// Angular distance between measurements (rad)
        pub angle_increment: f32,
        /// Time between measurements (s)
        pub time_increment: f32,
        /// Time between scans (s)
        pub scan_time: f32,
        /// Minimum range value (m)
        pub range_min: f32,
        /// Maximum range value (m)
        pub range_max: f32,
        /// Range data (m)
        pub ranges: Vec<f32>,
        /// Intensity data (device specific units)
        pub intensities: Vec<f32>,
    }
}

use msg::{Float32, Float32MultiArray, Float64, Float64MultiArray, Imu, LaserScan};

/// Quality of Service profile for ROS2 subscriptions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QosProfile {
    /// Best effort delivery (UDP-like)
    SensorData,
    /// Reliable delivery (TCP-like)
    SystemDefault,
    /// Parameter events
    Parameters,
    /// Services
    ServicesDefault,
}

impl QosProfile {
    /// Convert to the DDS QoS policy set used by RustDDS, mirroring the
    /// corresponding `rmw_qos_profile_*` defaults.
    pub fn to_qos(&self) -> QosPolicies {
        let builder = QosPolicyBuilder::new()
            .durability(policy::Durability::Volatile)
            .deadline(policy::Deadline(DdsDuration::INFINITE))
            .ownership(policy::Ownership::Shared)
            .lifespan(policy::Lifespan {
                duration: DdsDuration::INFINITE,
            });

        match self {
            QosProfile::SensorData => builder
                .reliability(policy::Reliability::BestEffort)
                .history(policy::History::KeepLast { depth: 5 })
                .build(),
            QosProfile::SystemDefault => builder
                .reliability(policy::Reliability::Reliable {
                    max_blocking_time: DdsDuration::from_millis(100),
                })
                .history(policy::History::KeepLast { depth: 10 })
                .build(),
            QosProfile::Parameters => builder
                .reliability(policy::Reliability::Reliable {
                    max_blocking_time: DdsDuration::from_millis(100),
                })
                .history(policy::History::KeepLast { depth: 1000 })
                .build(),
            QosProfile::ServicesDefault => builder
                .reliability(policy::Reliability::Reliable {
                    max_blocking_time: DdsDuration::from_millis(100),
                })
                .history(policy::History::KeepLast { depth: 10 })
                .build(),
        }
    }
}

/// Message type enum for common ROS2 message types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ros2MessageType {
    /// std_msgs/Float32
    Float32,
    /// std_msgs/Float64
    Float64,
    /// std_msgs/Float32MultiArray
    Float32Array,
    /// std_msgs/Float64MultiArray
    Float64Array,
    /// sensor_msgs/Imu (outputs 6-DOF: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
    Imu,
    /// sensor_msgs/LaserScan (outputs the `ranges` array)
    LaserScan,
    /// Custom message type.
    ///
    /// A custom type cannot be decoded without its IDL definition, so
    /// [`Ros2Stream::start`] rejects it with [`IoError::Unsupported`] rather
    /// than pretending to subscribe.
    Custom(String),
}

impl Ros2MessageType {
    /// The ROS2 interface name (`package/msg/Type`) this variant subscribes to.
    fn type_name(&self) -> Option<MessageTypeName> {
        match self {
            Ros2MessageType::Float32 => Some(MessageTypeName::new("std_msgs", "Float32")),
            Ros2MessageType::Float64 => Some(MessageTypeName::new("std_msgs", "Float64")),
            Ros2MessageType::Float32Array => {
                Some(MessageTypeName::new("std_msgs", "Float32MultiArray"))
            }
            Ros2MessageType::Float64Array => {
                Some(MessageTypeName::new("std_msgs", "Float64MultiArray"))
            }
            Ros2MessageType::Imu => Some(MessageTypeName::new("sensor_msgs", "Imu")),
            Ros2MessageType::LaserScan => Some(MessageTypeName::new("sensor_msgs", "LaserScan")),
            Ros2MessageType::Custom(_) => None,
        }
    }
}

/// Flatten an IMU message to the documented 6-DOF sample layout:
/// `[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]`.
pub fn imu_to_samples(imu: &Imu) -> [f32; 6] {
    [
        imu.linear_acceleration.x as f32,
        imu.linear_acceleration.y as f32,
        imu.linear_acceleration.z as f32,
        imu.angular_velocity.x as f32,
        imu.angular_velocity.y as f32,
        imu.angular_velocity.z as f32,
    ]
}

/// Configuration for ROS2 stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ros2Config {
    /// Topic name to subscribe to
    pub topic: String,

    /// Message type
    pub message_type: Ros2MessageType,

    /// Node name for this subscriber
    pub node_name: String,

    /// QoS profile
    pub qos: QosProfile,

    /// Buffer size for stream
    pub buffer_size: usize,

    /// Sample rate (Hz) - used for timing information
    pub sample_rate: f32,

    /// Number of channels in the output
    pub channels: usize,

    /// How long `read()` waits for a full block before reporting
    /// [`IoError::BufferEmpty`].
    pub read_timeout: Option<Duration>,
}

/// Build a ROS2-legal node base name from an arbitrary topic string.
///
/// ROS2 names admit only `[A-Za-z0-9_]`, must start with a letter or
/// underscore and must not contain `__`.
fn sanitize_node_name(topic: &str) -> String {
    let mut name = String::from("kizzasi_subscriber");
    let mut previous_underscore = true; // the seed already ends with one
    for ch in topic.chars() {
        if ch.is_ascii_alphanumeric() {
            name.push(ch);
            previous_underscore = false;
        } else if !previous_underscore {
            name.push('_');
            previous_underscore = true;
        }
    }
    while name.ends_with('_') {
        name.pop();
    }
    name
}

impl Ros2Config {
    /// Create a new ROS2 configuration
    pub fn new(topic: impl Into<String>, message_type: Ros2MessageType) -> Self {
        let topic = topic.into();
        let node_name = sanitize_node_name(&topic);

        Self {
            topic,
            message_type,
            node_name,
            qos: QosProfile::SensorData,
            buffer_size: 1024,
            sample_rate: 100.0, // Default 100 Hz
            channels: 1,
            read_timeout: Some(Duration::from_secs(1)),
        }
    }

    /// Set QoS profile
    pub fn with_qos(mut self, qos: QosProfile) -> Self {
        self.qos = qos;
        self
    }

    /// Set buffer size
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }

    /// Set sample rate
    pub fn with_sample_rate(mut self, sample_rate: f32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set number of channels
    pub fn with_channels(mut self, channels: usize) -> Self {
        self.channels = channels;
        self
    }

    /// Set how long `read()` waits for a full block
    pub fn with_read_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.read_timeout = timeout;
        self
    }

    /// The [`StreamConfig`] this ROS2 configuration describes
    pub fn stream_config(&self) -> StreamConfig {
        StreamConfig {
            sample_rate: self.sample_rate,
            channels: self.channels,
            buffer_size: self.buffer_size,
            timeout: self.read_timeout,
        }
    }
}

impl Default for Ros2Config {
    fn default() -> Self {
        Self::new("/sensor_data", Ros2MessageType::Float32)
    }
}

/// A live subscription, one variant per supported message type.
enum Ros2Subscription {
    Float32(Subscription<Float32>),
    Float64(Subscription<Float64>),
    Float32Array(Subscription<Float32MultiArray>),
    Float64Array(Subscription<Float64MultiArray>),
    Imu(Subscription<Imu>),
    LaserScan(Subscription<LaserScan>),
}

/// Take every message currently queued on `subscription`, appending the
/// decoded samples to `out`. Returns the number of messages consumed.
fn drain_subscription<M>(
    subscription: &Subscription<M>,
    out: &mut Vec<f32>,
    mut convert: impl FnMut(&M, &mut Vec<f32>),
) -> IoResult<usize>
where
    M: 'static + serde::de::DeserializeOwned,
{
    let mut count = 0usize;
    loop {
        match subscription.take() {
            Ok(Some((message, _info))) => {
                convert(&message, out);
                count += 1;
            }
            Ok(None) => return Ok(count),
            Err(e) => {
                return Err(IoError::Protocol(format!(
                    "ROS2 subscription read failed: {e}"
                )))
            }
        }
    }
}

/// Await the next message on `subscription` and append its samples to `out`.
async fn await_subscription<M>(
    subscription: &Subscription<M>,
    out: &mut Vec<f32>,
    mut convert: impl FnMut(&M, &mut Vec<f32>),
) -> IoResult<()>
where
    M: 'static + serde::de::DeserializeOwned,
{
    match subscription.async_take().await {
        Ok((message, _info)) => {
            convert(&message, out);
            Ok(())
        }
        Err(e) => Err(IoError::Protocol(format!(
            "ROS2 subscription read failed: {e}"
        ))),
    }
}

impl Ros2Subscription {
    /// Non-blocking drain of everything already received.
    fn drain(&self, out: &mut Vec<f32>) -> IoResult<usize> {
        match self {
            Ros2Subscription::Float32(sub) => drain_subscription(sub, out, |m, o| o.push(m.data)),
            Ros2Subscription::Float64(sub) => {
                drain_subscription(sub, out, |m, o| o.push(m.data as f32))
            }
            Ros2Subscription::Float32Array(sub) => {
                drain_subscription(sub, out, |m, o| o.extend_from_slice(&m.data))
            }
            Ros2Subscription::Float64Array(sub) => {
                drain_subscription(sub, out, |m, o| o.extend(m.data.iter().map(|v| *v as f32)))
            }
            Ros2Subscription::Imu(sub) => {
                drain_subscription(sub, out, |m, o| o.extend_from_slice(&imu_to_samples(m)))
            }
            Ros2Subscription::LaserScan(sub) => {
                drain_subscription(sub, out, |m, o| o.extend_from_slice(&m.ranges))
            }
        }
    }

    /// Await the next message.
    async fn await_next(&self, out: &mut Vec<f32>) -> IoResult<()> {
        match self {
            Ros2Subscription::Float32(sub) => {
                await_subscription(sub, out, |m, o| o.push(m.data)).await
            }
            Ros2Subscription::Float64(sub) => {
                await_subscription(sub, out, |m, o| o.push(m.data as f32)).await
            }
            Ros2Subscription::Float32Array(sub) => {
                await_subscription(sub, out, |m, o| o.extend_from_slice(&m.data)).await
            }
            Ros2Subscription::Float64Array(sub) => {
                await_subscription(sub, out, |m, o| o.extend(m.data.iter().map(|v| *v as f32)))
                    .await
            }
            Ros2Subscription::Imu(sub) => {
                await_subscription(sub, out, |m, o| o.extend_from_slice(&imu_to_samples(m))).await
            }
            Ros2Subscription::LaserScan(sub) => {
                await_subscription(sub, out, |m, o| o.extend_from_slice(&m.ranges)).await
            }
        }
    }
}

/// ROS2 subscriber stream
///
/// Subscribes to a ROS2 topic and converts messages to signal arrays.
pub struct Ros2Stream {
    config: Ros2Config,
    stream_config: StreamConfig,
    node: Node,
    subscription: Option<Ros2Subscription>,
    buffer: Vec<f32>,
    active: Arc<AtomicBool>,
}

impl Ros2Stream {
    /// Create a new ROS2 stream (creates the DDS participant and node).
    ///
    /// Call [`Ros2Stream::start`] afterwards to create the subscription; until
    /// then `read()` reports [`IoError::NotConnected`].
    pub async fn new(config: Ros2Config) -> IoResult<Self> {
        let context = Context::new()
            .map_err(|e| IoError::Connection(format!("Failed to create ROS2 context: {e}")))?;

        let node_name = NodeName::new("/", &config.node_name).map_err(|e| {
            IoError::ConfigError(format!(
                "Invalid ROS2 node name '{}': {e}",
                config.node_name
            ))
        })?;

        let node = context
            .new_node(node_name, NodeOptions::new().enable_rosout(false))
            .map_err(|e| IoError::Connection(format!("Failed to create ROS2 node: {e}")))?;

        let stream_config = config.stream_config();

        Ok(Self {
            config,
            stream_config,
            node,
            subscription: None,
            buffer: Vec::new(),
            active: Arc::new(AtomicBool::new(true)),
        })
    }

    /// Start the subscription for the configured message type.
    ///
    /// The created [`Subscription`] is stored on the stream. The previous
    /// implementation bound it to `_sub`, so it was dropped -- and the ROS2
    /// subscription torn down -- before `start()` even returned.
    pub async fn start(&mut self) -> IoResult<()> {
        let type_name = self.config.message_type.type_name().ok_or_else(|| {
            IoError::Unsupported(format!(
                "Message type {:?} cannot be decoded without its IDL definition",
                self.config.message_type
            ))
        })?;

        let name = Name::parse(&self.config.topic).map_err(|e| {
            IoError::ConfigError(format!("Invalid ROS2 topic '{}': {e}", self.config.topic))
        })?;
        let qos = self.config.qos.to_qos();

        let topic = self
            .node
            .create_topic(&name, type_name, &qos)
            .map_err(|e| IoError::Protocol(format!("Failed to create ROS2 topic: {e}")))?;

        let subscription = match self.config.message_type {
            Ros2MessageType::Float32 => Ros2Subscription::Float32(
                self.node
                    .create_subscription::<Float32>(&topic, Some(qos))
                    .map_err(subscribe_error)?,
            ),
            Ros2MessageType::Float64 => Ros2Subscription::Float64(
                self.node
                    .create_subscription::<Float64>(&topic, Some(qos))
                    .map_err(subscribe_error)?,
            ),
            Ros2MessageType::Float32Array => Ros2Subscription::Float32Array(
                self.node
                    .create_subscription::<Float32MultiArray>(&topic, Some(qos))
                    .map_err(subscribe_error)?,
            ),
            Ros2MessageType::Float64Array => Ros2Subscription::Float64Array(
                self.node
                    .create_subscription::<Float64MultiArray>(&topic, Some(qos))
                    .map_err(subscribe_error)?,
            ),
            Ros2MessageType::Imu => Ros2Subscription::Imu(
                self.node
                    .create_subscription::<Imu>(&topic, Some(qos))
                    .map_err(subscribe_error)?,
            ),
            Ros2MessageType::LaserScan => Ros2Subscription::LaserScan(
                self.node
                    .create_subscription::<LaserScan>(&topic, Some(qos))
                    .map_err(subscribe_error)?,
            ),
            Ros2MessageType::Custom(ref name) => {
                return Err(IoError::Unsupported(format!(
                    "Custom message type '{name}' cannot be decoded without its IDL definition"
                )))
            }
        };

        self.subscription = Some(subscription);
        self.active.store(true, Ordering::Release);
        Ok(())
    }

    /// Get the stream configuration
    pub fn stream_config(&self) -> StreamConfig {
        self.stream_config.clone()
    }

    /// Number of samples currently buffered but not yet returned by `read()`
    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }
}

fn subscribe_error(error: ros2_client::ros2::CreateError) -> IoError {
    IoError::Protocol(format!("Failed to subscribe: {error}"))
}

#[async_trait]
impl AsyncSignalStream for Ros2Stream {
    async fn read(&mut self) -> IoResult<Array1<f32>> {
        let block = self.stream_config.buffer_size.max(1);
        let timeout = self.stream_config.timeout;

        let Self {
            subscription,
            buffer,
            active,
            ..
        } = self;

        let subscription = subscription.as_ref().ok_or_else(|| {
            IoError::NotConnected(
                "Ros2Stream::start() must be called before reading a ROS2 topic".to_string(),
            )
        })?;

        // Consume whatever already arrived.
        subscription.drain(buffer)?;

        // Wait for the rest of the block, if the caller allows waiting.
        if buffer.len() < block && active.load(Ordering::Acquire) {
            if let Some(deadline) = timeout {
                let _ = tokio::time::timeout(deadline, async {
                    while buffer.len() < block {
                        subscription.await_next(buffer).await?;
                        subscription.drain(buffer)?;
                    }
                    Ok::<(), IoError>(())
                })
                .await;
            }
        }

        let still_active = active.load(Ordering::Acquire);
        if buffer.is_empty() {
            return Err(if still_active {
                IoError::BufferEmpty
            } else {
                IoError::EndOfStream
            });
        }
        if buffer.len() < block && still_active {
            // Partial block: keep the samples buffered rather than padding
            // them out with fabricated zeros.
            return Err(IoError::BufferEmpty);
        }

        let take = block.min(buffer.len());
        let data: Vec<f32> = buffer.drain(..take).collect();
        Ok(Array1::from_vec(data))
    }

    fn is_active(&self) -> bool {
        // An AtomicBool, so this sync method no longer blocks an async worker
        // thread inside `futures::executor::block_on`.
        self.active.load(Ordering::Acquire)
    }

    fn config(&self) -> &StreamConfig {
        // The real, user-configured StreamConfig -- a previous version
        // returned a `static DEFAULT_CONFIG` (100 Hz / 1 channel / 1024
        // samples) and discarded whatever with_sample_rate/with_channels/
        // with_buffer_size had set.
        &self.stream_config
    }

    async fn close(&mut self) -> IoResult<()> {
        self.active.store(false, Ordering::Release);
        self.subscription = None;
        Ok(())
    }
}

impl std::fmt::Debug for Ros2Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ros2Stream")
            .field("config", &self.config)
            .field("subscribed", &self.subscription.is_some())
            .field("buffered_samples", &self.buffer.len())
            .field("active", &self.is_active())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: no test constructs a `Context`: that starts a DDS domain
    // participant and multicast discovery, which is not available in every
    // build environment. The message decoding and configuration logic is
    // therefore factored into pure functions that are tested directly.

    #[test]
    fn test_ros2_config_creation() {
        let config = Ros2Config::new("/imu/data", Ros2MessageType::Imu);

        assert_eq!(config.topic, "/imu/data");
        assert_eq!(config.message_type, Ros2MessageType::Imu);
        assert_eq!(config.qos, QosProfile::SensorData);
        assert!(config.node_name.contains("imu_data"));
    }

    #[test]
    fn test_ros2_config_builder() {
        let config = Ros2Config::new("/laser", Ros2MessageType::LaserScan)
            .with_qos(QosProfile::SystemDefault)
            .with_buffer_size(2048)
            .with_sample_rate(50.0)
            .with_channels(2);

        assert_eq!(config.buffer_size, 2048);
        assert_eq!(config.sample_rate, 50.0);
        assert_eq!(config.channels, 2);
        assert_eq!(config.qos, QosProfile::SystemDefault);
    }

    #[test]
    fn test_stream_config_reflects_the_user_configuration() {
        // Regression: AsyncSignalStream::config() used to return a hardcoded
        // static (100 Hz / 1 channel / 1024 samples) regardless of the
        // builder calls.
        let config = Ros2Config::new("/imu/data", Ros2MessageType::Imu)
            .with_buffer_size(64)
            .with_sample_rate(400.0)
            .with_channels(6);
        let stream_config = config.stream_config();

        assert_eq!(stream_config.buffer_size, 64);
        assert_eq!(stream_config.sample_rate, 400.0);
        assert_eq!(stream_config.channels, 6);
    }

    #[test]
    fn test_node_names_are_ros2_legal() {
        // "/imu/data" used to become "kizzasi_subscriber__imu_data", which
        // ROS2 rejects (a double underscore is not a legal name).
        for topic in ["/imu/data", "imu", "/a//b/", "/scan"] {
            let name = sanitize_node_name(topic);
            assert!(!name.contains("__"), "{topic} -> {name}");
            assert!(!name.ends_with('_'), "{topic} -> {name}");
            assert!(
                name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_'),
                "{topic} -> {name}"
            );
            assert!(
                name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
                "{topic} -> {name}"
            );
            assert!(NodeName::new("/", &name).is_ok(), "{topic} -> {name}");
        }
    }

    #[test]
    fn test_message_types() {
        let msg_types = vec![
            Ros2MessageType::Float32,
            Ros2MessageType::Float64,
            Ros2MessageType::Imu,
            Ros2MessageType::LaserScan,
        ];

        for msg_type in msg_types {
            let config = Ros2Config::new("/test", msg_type.clone());
            assert_eq!(config.message_type, msg_type);
        }
    }

    #[test]
    fn test_every_non_custom_message_type_has_a_ros2_interface_name() {
        // LaserScan and Float64Array used to fall through to Unsupported at
        // start() despite being public enum variants.
        for msg_type in [
            Ros2MessageType::Float32,
            Ros2MessageType::Float64,
            Ros2MessageType::Float32Array,
            Ros2MessageType::Float64Array,
            Ros2MessageType::Imu,
            Ros2MessageType::LaserScan,
        ] {
            assert!(
                msg_type.type_name().is_some(),
                "{msg_type:?} has no interface name"
            );
        }
        assert!(Ros2MessageType::Custom("pkg/Msg".into())
            .type_name()
            .is_none());
    }

    #[test]
    fn test_imu_flattens_to_the_documented_six_dof_layout() {
        let imu = Imu {
            linear_acceleration: msg::Vector3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            angular_velocity: msg::Vector3 {
                x: 4.0,
                y: 5.0,
                z: 6.0,
            },
            ..Imu::default()
        };

        assert_eq!(imu_to_samples(&imu), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_qos_profiles_differ_in_reliability_and_history() {
        let sensor = QosProfile::SensorData.to_qos();
        let system = QosProfile::SystemDefault.to_qos();

        assert!(matches!(
            sensor.reliability(),
            Some(policy::Reliability::BestEffort)
        ));
        assert!(matches!(
            system.reliability(),
            Some(policy::Reliability::Reliable { .. })
        ));
        assert!(matches!(
            QosProfile::Parameters.to_qos().history(),
            Some(policy::History::KeepLast { depth: 1000 })
        ));
    }
}
