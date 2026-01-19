//! Serial port stream support
//!
//! Provides serial port communication for sensor and device interfacing.
//!
//! ## Features
//! - Async serial I/O with tokio-serial
//! - Configurable baud rate, parity, stop bits
//! - Line-based and raw binary reading
//! - Hardware flow control
//!
//! ## Example
//! ```rust,no_run
//! use kizzasi_io::{SerialStream, SerialConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = SerialConfig {
//!         port: "/dev/ttyUSB0".to_string(),
//!         baud_rate: 115200,
//!         ..Default::default()
//!     };
//!
//!     let mut stream = SerialStream::open(config)?;
//!
//!     while let Some(data) = stream.read_line().await? {
//!         println!("Received: {}", data);
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::error::{IoError, IoResult};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio_serial::{
    DataBits as TokioDataBits, FlowControl as TokioFlowControl, Parity as TokioParity, SerialPort,
    SerialPortBuilderExt, StopBits as TokioStopBits,
};
use tracing::{debug, info};

/// Serial port configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerialConfig {
    /// Serial port path (e.g., "/dev/ttyUSB0" or "COM3")
    pub port: String,

    /// Baud rate
    #[serde(default = "default_baud_rate")]
    pub baud_rate: u32,

    /// Data bits
    #[serde(default)]
    pub data_bits: DataBits,

    /// Parity
    #[serde(default)]
    pub parity: Parity,

    /// Stop bits
    #[serde(default)]
    pub stop_bits: StopBits,

    /// Flow control
    #[serde(default)]
    pub flow_control: FlowControl,

    /// Read timeout (ms)
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,

    /// Buffer size
    #[serde(default = "default_buffer_size")]
    pub buffer_size: usize,
}

fn default_baud_rate() -> u32 {
    115200
}

fn default_timeout() -> u64 {
    1000
}

fn default_buffer_size() -> usize {
    8192
}

impl Default for SerialConfig {
    fn default() -> Self {
        Self {
            port: String::new(),
            baud_rate: 115200,
            data_bits: DataBits::Eight,
            parity: Parity::None,
            stop_bits: StopBits::One,
            flow_control: FlowControl::None,
            timeout_ms: 1000,
            buffer_size: 8192,
        }
    }
}

/// Data bits
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum DataBits {
    Five,
    Six,
    Seven,
    #[default]
    Eight,
}

impl From<DataBits> for TokioDataBits {
    fn from(value: DataBits) -> Self {
        match value {
            DataBits::Five => TokioDataBits::Five,
            DataBits::Six => TokioDataBits::Six,
            DataBits::Seven => TokioDataBits::Seven,
            DataBits::Eight => TokioDataBits::Eight,
        }
    }
}

/// Parity
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum Parity {
    #[default]
    None,
    Odd,
    Even,
}

impl From<Parity> for TokioParity {
    fn from(value: Parity) -> Self {
        match value {
            Parity::None => TokioParity::None,
            Parity::Odd => TokioParity::Odd,
            Parity::Even => TokioParity::Even,
        }
    }
}

/// Stop bits
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum StopBits {
    #[default]
    One,
    Two,
}

impl From<StopBits> for TokioStopBits {
    fn from(value: StopBits) -> Self {
        match value {
            StopBits::One => TokioStopBits::One,
            StopBits::Two => TokioStopBits::Two,
        }
    }
}

/// Flow control
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum FlowControl {
    #[default]
    None,
    Software,
    Hardware,
}

impl From<FlowControl> for TokioFlowControl {
    fn from(value: FlowControl) -> Self {
        match value {
            FlowControl::None => TokioFlowControl::None,
            FlowControl::Software => TokioFlowControl::Software,
            FlowControl::Hardware => TokioFlowControl::Hardware,
        }
    }
}

/// Serial port stream
pub struct SerialStream {
    config: SerialConfig,
    port: tokio_serial::SerialStream,
    reader: BufReader<tokio_serial::SerialStream>,
}

impl SerialStream {
    /// Open serial port
    pub fn open(config: SerialConfig) -> IoResult<Self> {
        let port = tokio_serial::new(&config.port, config.baud_rate)
            .data_bits(config.data_bits.into())
            .parity(config.parity.into())
            .stop_bits(config.stop_bits.into())
            .flow_control(config.flow_control.into())
            .timeout(Duration::from_millis(config.timeout_ms))
            .open_native_async()
            .map_err(|e| IoError::ConnectionFailed(format!("Failed to open serial port: {}", e)))?;

        info!("Serial port opened: {}", config.port);

        // Clone port for reader
        let reader_port = tokio_serial::new(&config.port, config.baud_rate)
            .data_bits(config.data_bits.into())
            .parity(config.parity.into())
            .stop_bits(config.stop_bits.into())
            .flow_control(config.flow_control.into())
            .timeout(Duration::from_millis(config.timeout_ms))
            .open_native_async()
            .map_err(|e| IoError::ConnectionFailed(format!("Failed to open serial port: {}", e)))?;

        let reader = BufReader::with_capacity(config.buffer_size, reader_port);

        Ok(Self {
            config,
            port,
            reader,
        })
    }

    /// Read raw bytes
    pub async fn read_bytes(&mut self, max_len: usize) -> IoResult<Bytes> {
        let mut buffer = vec![0u8; max_len];
        match self.port.read(&mut buffer).await {
            Ok(n) => {
                debug!("Serial read {} bytes", n);
                Ok(Bytes::from(buffer[..n].to_vec()))
            }
            Err(e) => Err(IoError::ReadFailed(format!("Serial read error: {}", e))),
        }
    }

    /// Read line (terminated by \n)
    pub async fn read_line(&mut self) -> IoResult<Option<String>> {
        let mut line = String::new();
        match self.reader.read_line(&mut line).await {
            Ok(0) => {
                info!("Serial port closed");
                Ok(None)
            }
            Ok(_) => {
                let trimmed = line.trim_end().to_string();
                debug!("Serial read line: {}", trimmed);
                Ok(Some(trimmed))
            }
            Err(e) => Err(IoError::ReadFailed(format!(
                "Serial read line error: {}",
                e
            ))),
        }
    }

    /// Read until delimiter
    pub async fn read_until(&mut self, delimiter: u8) -> IoResult<Bytes> {
        let mut buffer = Vec::new();
        match self.reader.read_until(delimiter, &mut buffer).await {
            Ok(_) => {
                debug!(
                    "Serial read until 0x{:02x}: {} bytes",
                    delimiter,
                    buffer.len()
                );
                Ok(Bytes::from(buffer))
            }
            Err(e) => Err(IoError::ReadFailed(format!(
                "Serial read until error: {}",
                e
            ))),
        }
    }

    /// Write data
    pub async fn write(&mut self, data: &[u8]) -> IoResult<()> {
        self.port
            .write_all(data)
            .await
            .map_err(|e| IoError::SendFailed(format!("Serial write error: {}", e)))?;

        debug!("Serial wrote {} bytes", data.len());
        Ok(())
    }

    /// Write line (append \n)
    pub async fn write_line(&mut self, line: &str) -> IoResult<()> {
        let data = format!("{}\n", line);
        self.write(data.as_bytes()).await
    }

    /// Flush output buffer
    pub async fn flush(&mut self) -> IoResult<()> {
        self.port
            .flush()
            .await
            .map_err(|e| IoError::SendFailed(format!("Serial flush error: {}", e)))
    }

    /// Get the number of bytes available to read
    pub fn bytes_to_read(&self) -> IoResult<u32> {
        self.port
            .bytes_to_read()
            .map_err(|e| IoError::ReadFailed(format!("Failed to get bytes to read: {}", e)))
    }

    /// Get the number of bytes waiting to be written
    pub fn bytes_to_write(&self) -> IoResult<u32> {
        self.port
            .bytes_to_write()
            .map_err(|e| IoError::SendFailed(format!("Failed to get bytes to write: {}", e)))
    }

    /// Clear input buffer
    pub fn clear_input_buffer(&self) -> IoResult<()> {
        self.port
            .clear(tokio_serial::ClearBuffer::Input)
            .map_err(|e| IoError::ConfigError(format!("Failed to clear input buffer: {}", e)))
    }

    /// Clear output buffer
    pub fn clear_output_buffer(&self) -> IoResult<()> {
        self.port
            .clear(tokio_serial::ClearBuffer::Output)
            .map_err(|e| IoError::ConfigError(format!("Failed to clear output buffer: {}", e)))
    }

    /// Clear all buffers
    pub fn clear_all_buffers(&self) -> IoResult<()> {
        self.port
            .clear(tokio_serial::ClearBuffer::All)
            .map_err(|e| IoError::ConfigError(format!("Failed to clear buffers: {}", e)))
    }

    /// Set baud rate
    pub fn set_baud_rate(&mut self, baud_rate: u32) -> IoResult<()> {
        self.port
            .set_baud_rate(baud_rate)
            .map_err(|e| IoError::ConfigError(format!("Failed to set baud rate: {}", e)))?;

        self.config.baud_rate = baud_rate;
        info!("Serial baud rate set to {}", baud_rate);
        Ok(())
    }
}

/// List available serial ports
pub fn list_ports() -> IoResult<Vec<String>> {
    tokio_serial::available_ports()
        .map(|ports| ports.into_iter().map(|p| p.port_name).collect::<Vec<_>>())
        .map_err(|e| IoError::ConfigError(format!("Failed to list serial ports: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serial_config_defaults() {
        let config = SerialConfig::default();
        assert_eq!(config.baud_rate, 115200);
        assert_eq!(config.timeout_ms, 1000);
        assert_eq!(config.buffer_size, 8192);
    }

    #[test]
    fn test_serial_config_serialize() {
        let config = SerialConfig {
            port: "/dev/ttyUSB0".to_string(),
            baud_rate: 9600,
            data_bits: DataBits::Eight,
            parity: Parity::None,
            stop_bits: StopBits::One,
            flow_control: FlowControl::None,
            timeout_ms: 2000,
            buffer_size: 4096,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SerialConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.port, config.port);
        assert_eq!(deserialized.baud_rate, config.baud_rate);
    }

    #[test]
    fn test_list_ports() {
        // This test will succeed even if no ports are available
        let result = list_ports();
        assert!(result.is_ok());
    }
}
