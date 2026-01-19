//! Error types for kizzasi-io

use thiserror::Error;

/// Result type alias for IO operations
pub type IoResult<T> = Result<T, IoError>;

/// Errors that can occur in the IO module
#[derive(Error, Debug)]
pub enum IoError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Stream error: {0}")]
    StreamError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Signal processing error: {0}")]
    SignalError(String),

    #[error("Read failed: {0}")]
    ReadFailed(String),

    #[error("Write failed: {0}")]
    WriteFailed(String),

    #[error("Send failed: {0}")]
    SendFailed(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    #[error("Buffer is full")]
    BufferFull,

    #[error("Buffer is empty")]
    BufferEmpty,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Core error: {0}")]
    CoreError(#[from] kizzasi_core::CoreError),

    #[error("IO error: {0}")]
    StdIoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[cfg(feature = "mqtt")]
    #[error("MQTT error: {0}")]
    MqttError(#[from] rumqttc::ClientError),
}
