//! Stream recording and playback
//!
//! Provides utilities for recording signal streams to files and playing them back.
//!
//! ## Features
//! - Multi-format recording (binary, JSON, CSV)
//! - Timestamp preservation
//! - Metadata support
//! - Streaming playback
//! - Frame-accurate timing
//!
//! ## Example
//! ```rust,no_run
//! use kizzasi_io::{StreamRecorder, StreamPlayer, RecorderConfig, RecorderFormat};
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Record
//!     let config = RecorderConfig {
//!         path: "recording.bin".to_string(),
//!         format: RecorderFormat::Binary,
//!         buffer_size: 1024,
//!         ..Default::default()
//!     };
//!
//!     let mut recorder = StreamRecorder::new(config).await?;
//!
//!     // Record some samples
//!     recorder.record_samples(&[1.0, 2.0, 3.0], None).await?;
//!     recorder.finalize().await?;
//!
//!     // Playback
//!     let mut player = StreamPlayer::new("recording.bin").await?;
//!     while let Some(frame) = player.next_frame().await? {
//!         println!("Samples: {:?}", frame.samples);
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::error::{IoError, IoResult};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, SystemTime};
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tracing::{debug, info};

/// Recording format
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum RecorderFormat {
    /// Binary format (most efficient)
    #[default]
    Binary,
    /// JSON format (human-readable)
    Json,
    /// CSV format (spreadsheet-compatible)
    Csv,
}

/// Recorder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecorderConfig {
    /// File path
    pub path: String,

    /// Recording format
    #[serde(default)]
    pub format: RecorderFormat,

    /// Sample rate (Hz)
    #[serde(default = "default_sample_rate")]
    pub sample_rate: f32,

    /// Number of channels
    #[serde(default = "default_channels")]
    pub channels: usize,

    /// Buffer size
    #[serde(default = "default_buffer_size")]
    pub buffer_size: usize,

    /// Record timestamps
    #[serde(default = "default_true")]
    pub record_timestamps: bool,

    /// Metadata
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

fn default_sample_rate() -> f32 {
    44100.0
}

fn default_channels() -> usize {
    1
}

fn default_buffer_size() -> usize {
    1024
}

fn default_true() -> bool {
    true
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            path: String::new(),
            format: RecorderFormat::Binary,
            sample_rate: 44100.0,
            channels: 1,
            buffer_size: 1024,
            record_timestamps: true,
            metadata: std::collections::HashMap::new(),
        }
    }
}

/// Recorded frame with optional timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedFrame {
    /// Sample data
    pub samples: Vec<f32>,

    /// Timestamp (seconds since start)
    pub timestamp: Option<f64>,

    /// Frame number
    pub frame_number: usize,
}

/// Stream recorder
pub struct StreamRecorder {
    config: RecorderConfig,
    writer: BufWriter<File>,
    frame_count: usize,
    start_time: SystemTime,
    total_samples: usize,
}

impl StreamRecorder {
    /// Create a new stream recorder
    pub async fn new(config: RecorderConfig) -> IoResult<Self> {
        let file = File::create(&config.path)
            .await
            .map_err(|e| IoError::WriteFailed(format!("Failed to create recording: {}", e)))?;

        let mut writer = BufWriter::new(file);

        // Write header based on format
        match config.format {
            RecorderFormat::Binary => {
                // Write magic number
                writer
                    .write_all(b"ZHREC001")
                    .await
                    .map_err(|e| IoError::WriteFailed(format!("Failed to write header: {}", e)))?;

                // Write config
                let config_json = serde_json::to_string(&config).map_err(|e| {
                    IoError::WriteFailed(format!("Failed to serialize config: {}", e))
                })?;
                let config_len = config_json.len() as u32;
                writer
                    .write_all(&config_len.to_le_bytes())
                    .await
                    .map_err(|e| {
                        IoError::WriteFailed(format!("Failed to write config length: {}", e))
                    })?;
                writer
                    .write_all(config_json.as_bytes())
                    .await
                    .map_err(|e| IoError::WriteFailed(format!("Failed to write config: {}", e)))?;
            }
            RecorderFormat::Json => {
                // Write metadata header
                let header = serde_json::json!({
                    "format": "kizzasi-io-recording",
                    "version": "1.0",
                    "config": config,
                    "frames": []
                });
                let header_str = serde_json::to_string_pretty(&header).map_err(|e| {
                    IoError::WriteFailed(format!("Failed to write JSON header: {}", e))
                })?;
                writer
                    .write_all(header_str.as_bytes())
                    .await
                    .map_err(|e| IoError::WriteFailed(format!("Failed to write header: {}", e)))?;
            }
            RecorderFormat::Csv => {
                // Write CSV header
                let header = if config.record_timestamps {
                    "frame,timestamp,samples\n"
                } else {
                    "frame,samples\n"
                };
                writer.write_all(header.as_bytes()).await.map_err(|e| {
                    IoError::WriteFailed(format!("Failed to write CSV header: {}", e))
                })?;
            }
        }

        info!("Stream recorder created: {:?}", config.path);

        Ok(Self {
            config,
            writer,
            frame_count: 0,
            start_time: SystemTime::now(),
            total_samples: 0,
        })
    }

    /// Record samples
    pub async fn record_samples(
        &mut self,
        samples: &[f32],
        timestamp: Option<f64>,
    ) -> IoResult<()> {
        let frame = RecordedFrame {
            samples: samples.to_vec(),
            timestamp: timestamp.or_else(|| {
                if self.config.record_timestamps {
                    Some(
                        self.start_time
                            .elapsed()
                            .unwrap_or(Duration::ZERO)
                            .as_secs_f64(),
                    )
                } else {
                    None
                }
            }),
            frame_number: self.frame_count,
        };

        self.write_frame(&frame).await?;
        self.frame_count += 1;
        self.total_samples += samples.len();

        Ok(())
    }

    /// Record an array of samples
    pub async fn record_array(&mut self, samples: &Array1<f32>) -> IoResult<()> {
        let vec: Vec<f32> = samples.to_vec();
        self.record_samples(&vec, None).await
    }

    /// Write a frame to the file
    async fn write_frame(&mut self, frame: &RecordedFrame) -> IoResult<()> {
        match self.config.format {
            RecorderFormat::Binary => {
                // Write frame header: sample count (u32) + optional timestamp (f64)
                let sample_count = frame.samples.len() as u32;
                self.writer
                    .write_all(&sample_count.to_le_bytes())
                    .await
                    .map_err(|e| IoError::WriteFailed(format!("Failed to write frame: {}", e)))?;

                if let Some(ts) = frame.timestamp {
                    self.writer
                        .write_all(&ts.to_le_bytes())
                        .await
                        .map_err(|e| {
                            IoError::WriteFailed(format!("Failed to write timestamp: {}", e))
                        })?;
                }

                // Write samples
                for &sample in &frame.samples {
                    self.writer
                        .write_all(&sample.to_le_bytes())
                        .await
                        .map_err(|e| {
                            IoError::WriteFailed(format!("Failed to write sample: {}", e))
                        })?;
                }
            }
            RecorderFormat::Json => {
                let json = serde_json::to_string(&frame).map_err(|e| {
                    IoError::WriteFailed(format!("Failed to serialize frame: {}", e))
                })?;
                self.writer.write_all(json.as_bytes()).await.map_err(|e| {
                    IoError::WriteFailed(format!("Failed to write JSON frame: {}", e))
                })?;
                self.writer
                    .write_all(b"\n")
                    .await
                    .map_err(|e| IoError::WriteFailed(format!("Failed to write newline: {}", e)))?;
            }
            RecorderFormat::Csv => {
                let samples_str = frame
                    .samples
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(";");

                let line = if let Some(ts) = frame.timestamp {
                    format!("{},{},{}\n", frame.frame_number, ts, samples_str)
                } else {
                    format!("{},{}\n", frame.frame_number, samples_str)
                };

                self.writer.write_all(line.as_bytes()).await.map_err(|e| {
                    IoError::WriteFailed(format!("Failed to write CSV line: {}", e))
                })?;
            }
        }

        debug!(
            "Recorded frame {}: {} samples",
            frame.frame_number,
            frame.samples.len()
        );

        Ok(())
    }

    /// Flush and finalize recording
    pub async fn finalize(mut self) -> IoResult<()> {
        self.writer
            .flush()
            .await
            .map_err(|e| IoError::WriteFailed(format!("Failed to flush recording: {}", e)))?;

        info!(
            "Recording finalized: {} frames, {} samples",
            self.frame_count, self.total_samples
        );

        Ok(())
    }

    /// Get frame count
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Get total samples recorded
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Create a player for this recording
    pub async fn create_player(&self) -> IoResult<StreamPlayer> {
        StreamPlayer::new(&self.config.path).await
    }
}

/// Stream player for playback
pub struct StreamPlayer {
    config: RecorderConfig,
    reader: BufReader<File>,
    frame_count: usize,
    format: RecorderFormat,
}

impl StreamPlayer {
    /// Open a recording for playback
    pub async fn new<P: AsRef<Path>>(path: P) -> IoResult<Self> {
        let file = File::open(path)
            .await
            .map_err(|e| IoError::ReadFailed(format!("Failed to open recording: {}", e)))?;

        let mut reader = BufReader::new(file);

        // Read header to determine format
        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .await
            .map_err(|e| IoError::ReadFailed(format!("Failed to read magic: {}", e)))?;

        let (format, config) = if &magic == b"ZHREC001" {
            // Binary format
            let mut len_bytes = [0u8; 4];
            reader
                .read_exact(&mut len_bytes)
                .await
                .map_err(|e| IoError::ReadFailed(format!("Failed to read config length: {}", e)))?;
            let config_len = u32::from_le_bytes(len_bytes) as usize;

            let mut config_bytes = vec![0u8; config_len];
            reader
                .read_exact(&mut config_bytes)
                .await
                .map_err(|e| IoError::ReadFailed(format!("Failed to read config: {}", e)))?;

            let config: RecorderConfig = serde_json::from_slice(&config_bytes)
                .map_err(|e| IoError::ReadFailed(format!("Failed to parse config: {}", e)))?;

            (RecorderFormat::Binary, config)
        } else {
            // Try JSON or CSV
            return Err(IoError::ReadFailed(
                "Non-binary format playback not yet implemented".into(),
            ));
        };

        info!("Stream player opened: {:?}", format);

        Ok(Self {
            config,
            reader,
            frame_count: 0,
            format,
        })
    }

    /// Read next frame
    pub async fn next_frame(&mut self) -> IoResult<Option<RecordedFrame>> {
        match self.format {
            RecorderFormat::Binary => {
                // Read sample count
                let mut count_bytes = [0u8; 4];
                match self.reader.read_exact(&mut count_bytes).await {
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
                    Err(e) => {
                        return Err(IoError::ReadFailed(format!(
                            "Failed to read frame count: {}",
                            e
                        )))
                    }
                }

                let sample_count = u32::from_le_bytes(count_bytes) as usize;

                // Read timestamp if enabled
                let timestamp = if self.config.record_timestamps {
                    let mut ts_bytes = [0u8; 8];
                    self.reader.read_exact(&mut ts_bytes).await.map_err(|e| {
                        IoError::ReadFailed(format!("Failed to read timestamp: {}", e))
                    })?;
                    Some(f64::from_le_bytes(ts_bytes))
                } else {
                    None
                };

                // Read samples
                let mut samples = Vec::with_capacity(sample_count);
                for _ in 0..sample_count {
                    let mut sample_bytes = [0u8; 4];
                    self.reader
                        .read_exact(&mut sample_bytes)
                        .await
                        .map_err(|e| {
                            IoError::ReadFailed(format!("Failed to read sample: {}", e))
                        })?;
                    samples.push(f32::from_le_bytes(sample_bytes));
                }

                let frame = RecordedFrame {
                    samples,
                    timestamp,
                    frame_number: self.frame_count,
                };

                self.frame_count += 1;
                debug!("Read frame {}", frame.frame_number);

                Ok(Some(frame))
            }
            _ => Err(IoError::ReadFailed("Format not supported yet".into())),
        }
    }

    /// Seek to frame
    pub async fn seek_to_frame(&mut self, _frame_number: usize) -> IoResult<()> {
        // TODO: Implement seeking
        Err(IoError::ReadFailed("Seeking not yet implemented".into()))
    }

    /// Get configuration
    pub fn config(&self) -> &RecorderConfig {
        &self.config
    }

    /// Get current frame number
    pub fn frame_number(&self) -> usize {
        self.frame_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[tokio::test]
    async fn test_recorder_binary() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_recording.bin");

        let config = RecorderConfig {
            path: path.to_string_lossy().to_string(),
            format: RecorderFormat::Binary,
            sample_rate: 44100.0,
            channels: 1,
            buffer_size: 1024,
            record_timestamps: true,
            metadata: std::collections::HashMap::new(),
        };

        // Record
        let mut recorder = StreamRecorder::new(config).await.unwrap();
        recorder
            .record_samples(&[1.0, 2.0, 3.0], None)
            .await
            .unwrap();
        recorder
            .record_samples(&[4.0, 5.0, 6.0], None)
            .await
            .unwrap();
        recorder.finalize().await.unwrap();

        // Playback
        let mut player = StreamPlayer::new(&path).await.unwrap();

        let frame1 = player.next_frame().await.unwrap().unwrap();
        assert_eq!(frame1.samples, vec![1.0, 2.0, 3.0]);
        assert_eq!(frame1.frame_number, 0);

        let frame2 = player.next_frame().await.unwrap().unwrap();
        assert_eq!(frame2.samples, vec![4.0, 5.0, 6.0]);
        assert_eq!(frame2.frame_number, 1);

        assert!(player.next_frame().await.unwrap().is_none());

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn test_recorder_array() {
        let temp_dir = env::temp_dir();
        let path = temp_dir.join("test_recording_array.bin");

        let config = RecorderConfig {
            path: path.to_string_lossy().to_string(),
            format: RecorderFormat::Binary,
            sample_rate: 48000.0,
            channels: 2,
            buffer_size: 512,
            record_timestamps: false,
            metadata: std::collections::HashMap::new(),
        };

        let mut recorder = StreamRecorder::new(config).await.unwrap();

        let samples = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        recorder.record_array(&samples).await.unwrap();
        recorder.finalize().await.unwrap();

        // Cleanup
        std::fs::remove_file(path).ok();
    }
}
