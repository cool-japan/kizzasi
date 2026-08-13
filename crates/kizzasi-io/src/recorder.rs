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
use tracing::{debug, info, warn};

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
                // Terminate the header on its own line. `StreamPlayer::new` splits the file
                // by accumulating lines until the buffer parses as a standalone JSON value;
                // without this newline, the header's closing `}` glues onto the first JSONL
                // frame line (`}{"samples":...}`), which never parses as valid JSON on its
                // own, so the header is never recognized as complete.
                writer.write_all(b"\n").await.map_err(|e| {
                    IoError::WriteFailed(format!("Failed to write header terminator: {}", e))
                })?;
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
        // The binary frame layout has no per-frame flag marking timestamp
        // presence: `write_frame` writes 8 timestamp bytes iff
        // `frame.timestamp.is_some()`, and the reader decides whether to
        // consume those bytes purely from the recording header's
        // `record_timestamps` flag. So `frame.timestamp` MUST agree with
        // `self.config.record_timestamps` on every frame, regardless of
        // what the caller passed in, or the two ends desync: a caller
        // passing `Some(ts)` while `record_timestamps == false` used to
        // write 8 extra bytes per frame that the reader would then
        // misinterpret as the next frame's sample count and length.
        if timestamp.is_some() && !self.config.record_timestamps {
            warn!(
                "record_samples: explicit timestamp ignored because record_timestamps is false \
                 for this recording"
            );
        }

        let frame = RecordedFrame {
            samples: samples.to_vec(),
            timestamp: if self.config.record_timestamps {
                Some(timestamp.unwrap_or_else(|| {
                    self.start_time
                        .elapsed()
                        .unwrap_or(Duration::ZERO)
                        .as_secs_f64()
                }))
            } else {
                None
            },
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
    /// Pre-loaded frames for JSON/CSV formats
    text_frames: Vec<RecordedFrame>,
    /// Index into text_frames for the next frame to return
    text_frame_index: usize,
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
            // Try JSON or CSV — magic contains the first 8 bytes of the text file
            // Read the rest of the file
            let mut rest = Vec::new();
            reader
                .read_to_end(&mut rest)
                .await
                .map_err(|e| IoError::ReadFailed(format!("Failed to read file: {}", e)))?;

            // Reconstruct full file content
            let mut content = Vec::with_capacity(8 + rest.len());
            content.extend_from_slice(&magic);
            content.extend_from_slice(&rest);

            let content_str = String::from_utf8(content)
                .map_err(|e| IoError::ReadFailed(format!("File is not valid UTF-8: {}", e)))?;

            // Detect format by first non-whitespace byte
            let first_char = content_str.trim_start().chars().next().unwrap_or('\0');

            if first_char == '{' || first_char == '[' {
                // JSON format: first line is pretty-printed header, rest are JSONL frames
                let all_lines: Vec<&str> = content_str.lines().collect();
                let mut header_buf = String::new();
                let mut header_parsed: Option<serde_json::Value> = None;
                let mut remaining_lines: Vec<&str> = Vec::new();
                let mut i = 0;
                while i < all_lines.len() {
                    header_buf.push_str(all_lines[i]);
                    header_buf.push('\n');
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&header_buf) {
                        header_parsed = Some(v);
                        i += 1;
                        remaining_lines = all_lines[i..].to_vec();
                        break;
                    }
                    i += 1;
                }

                let header_val = header_parsed
                    .ok_or_else(|| IoError::ReadFailed("Failed to parse JSON header".into()))?;

                let config: RecorderConfig = serde_json::from_value(header_val["config"].clone())
                    .map_err(|e| {
                    IoError::ReadFailed(format!("Failed to parse config from JSON header: {}", e))
                })?;

                // Parse JSONL frames
                let mut text_frames: Vec<RecordedFrame> = Vec::new();
                for line in remaining_lines {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let frame: RecordedFrame = serde_json::from_str(trimmed).map_err(|e| {
                        IoError::ReadFailed(format!("Failed to parse JSON frame: {}", e))
                    })?;
                    text_frames.push(frame);
                }

                info!("Stream player opened: Json ({} frames)", text_frames.len());

                return Ok(Self {
                    config,
                    reader,
                    frame_count: 0,
                    format: RecorderFormat::Json,
                    text_frames,
                    text_frame_index: 0,
                });
            } else {
                // CSV format
                let mut csv_lines = content_str.lines();

                let header_line = csv_lines
                    .next()
                    .ok_or_else(|| IoError::ReadFailed("CSV file is empty".into()))?;

                // Detect whether timestamps are present
                let has_timestamp = header_line.contains("timestamp");

                let config = RecorderConfig {
                    record_timestamps: has_timestamp,
                    ..RecorderConfig::default()
                };

                let mut text_frames: Vec<RecordedFrame> = Vec::new();
                for (line_idx, line) in csv_lines.enumerate() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let (frame_number, timestamp, samples) = if has_timestamp {
                        let parts: Vec<&str> = trimmed.splitn(3, ',').collect();
                        if parts.len() < 3 {
                            return Err(IoError::ReadFailed(format!(
                                "CSV line {} has too few fields: {:?}",
                                line_idx + 2,
                                trimmed
                            )));
                        }
                        let fn_ = parts[0].parse::<usize>().map_err(|e| {
                            IoError::ReadFailed(format!(
                                "Invalid frame number on line {}: {}",
                                line_idx + 2,
                                e
                            ))
                        })?;
                        let ts = parts[1].parse::<f64>().map_err(|e| {
                            IoError::ReadFailed(format!(
                                "Invalid timestamp on line {}: {}",
                                line_idx + 2,
                                e
                            ))
                        })?;
                        let s: Vec<f32> = parts[2]
                            .split(';')
                            .filter(|s| !s.is_empty())
                            .map(|s| {
                                s.parse::<f32>().map_err(|e| {
                                    IoError::ReadFailed(format!(
                                        "Invalid sample value '{}': {}",
                                        s, e
                                    ))
                                })
                            })
                            .collect::<IoResult<Vec<f32>>>()?;
                        (fn_, Some(ts), s)
                    } else {
                        let parts: Vec<&str> = trimmed.splitn(2, ',').collect();
                        if parts.len() < 2 {
                            return Err(IoError::ReadFailed(format!(
                                "CSV line {} has too few fields: {:?}",
                                line_idx + 2,
                                trimmed
                            )));
                        }
                        let fn_ = parts[0].parse::<usize>().map_err(|e| {
                            IoError::ReadFailed(format!(
                                "Invalid frame number on line {}: {}",
                                line_idx + 2,
                                e
                            ))
                        })?;
                        let s: Vec<f32> = parts[1]
                            .split(';')
                            .filter(|s| !s.is_empty())
                            .map(|s| {
                                s.parse::<f32>().map_err(|e| {
                                    IoError::ReadFailed(format!(
                                        "Invalid sample value '{}': {}",
                                        s, e
                                    ))
                                })
                            })
                            .collect::<IoResult<Vec<f32>>>()?;
                        (fn_, None, s)
                    };
                    text_frames.push(RecordedFrame {
                        samples,
                        timestamp,
                        frame_number,
                    });
                }

                info!("Stream player opened: Csv ({} frames)", text_frames.len());

                return Ok(Self {
                    config,
                    reader,
                    frame_count: 0,
                    format: RecorderFormat::Csv,
                    text_frames,
                    text_frame_index: 0,
                });
            }
        };

        info!("Stream player opened: {:?}", format);

        Ok(Self {
            config,
            reader,
            frame_count: 0,
            format,
            text_frames: Vec::new(),
            text_frame_index: 0,
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
            RecorderFormat::Json | RecorderFormat::Csv => {
                if self.text_frame_index >= self.text_frames.len() {
                    return Ok(None);
                }
                let mut frame = self.text_frames[self.text_frame_index].clone();
                frame.frame_number = self.frame_count;
                self.text_frame_index += 1;
                self.frame_count += 1;
                debug!("Read frame {}", frame.frame_number);
                Ok(Some(frame))
            }
        }
    }

    /// Seek to frame
    pub async fn seek_to_frame(&mut self, frame_number: usize) -> IoResult<()> {
        use std::io::SeekFrom;
        use tokio::io::AsyncSeekExt;

        match self.format {
            RecorderFormat::Binary => {
                // Rewind to file start
                self.reader
                    .seek(SeekFrom::Start(0))
                    .await
                    .map_err(|e| IoError::ReadFailed(format!("Failed to seek to start: {}", e)))?;

                // Skip 8-byte magic ("ZHREC001")
                self.reader
                    .seek(SeekFrom::Current(8))
                    .await
                    .map_err(|e| IoError::ReadFailed(format!("Failed to skip magic: {}", e)))?;

                // Read config_len, then skip those bytes
                let mut len_bytes = [0u8; 4];
                self.reader.read_exact(&mut len_bytes).await.map_err(|e| {
                    IoError::ReadFailed(format!("Failed to read config length: {}", e))
                })?;
                let config_len = u32::from_le_bytes(len_bytes) as i64;
                self.reader
                    .seek(SeekFrom::Current(config_len))
                    .await
                    .map_err(|e| IoError::ReadFailed(format!("Failed to skip config: {}", e)))?;

                // Advance past frame_number frames
                let ts_bytes: i64 = if self.config.record_timestamps { 8 } else { 0 };
                for i in 0..frame_number {
                    let mut count_buf = [0u8; 4];
                    match self.reader.read_exact(&mut count_buf).await {
                        Ok(_) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                            return Err(IoError::ReadFailed(format!(
                                "Cannot seek to frame {}: only {} frames in recording",
                                frame_number, i
                            )));
                        }
                        Err(e) => {
                            return Err(IoError::ReadFailed(format!(
                                "Failed to read frame {} count: {}",
                                i, e
                            )));
                        }
                    }
                    let sample_count = u32::from_le_bytes(count_buf) as i64;
                    let frame_body = ts_bytes + sample_count * 4;
                    self.reader
                        .seek(SeekFrom::Current(frame_body))
                        .await
                        .map_err(|e| {
                            IoError::ReadFailed(format!("Failed to skip frame {}: {}", i, e))
                        })?;
                }

                self.frame_count = frame_number;
                Ok(())
            }
            _ => Err(IoError::ReadFailed(
                "Seeking is only supported for binary format".into(),
            )),
        }
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

    #[tokio::test]
    async fn test_seek_to_frame_binary() {
        let temp_dir = std::env::temp_dir();
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let path = temp_dir.join(format!(
            "kizzasi_seek_test_{}_{}.bin",
            std::process::id(),
            id
        ));

        let config = RecorderConfig {
            path: path.to_string_lossy().to_string(),
            format: RecorderFormat::Binary,
            sample_rate: 44100.0,
            channels: 1,
            buffer_size: 1024,
            record_timestamps: true,
            metadata: std::collections::HashMap::new(),
        };

        let mut recorder = StreamRecorder::new(config).await.unwrap();
        for i in 0usize..5 {
            let v = i as f32;
            recorder
                .record_samples(&[v, v + 1.0, v + 2.0], None)
                .await
                .unwrap();
        }
        recorder.finalize().await.unwrap();

        let mut player = StreamPlayer::new(&path).await.unwrap();
        player.seek_to_frame(2).await.unwrap();
        let frame = player.next_frame().await.unwrap().unwrap();
        assert_eq!(frame.frame_number, 2);
        assert_eq!(frame.samples, vec![2.0, 3.0, 4.0]);

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn test_seek_to_frame_zero() {
        let temp_dir = std::env::temp_dir();
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(1);
        let path = temp_dir.join(format!(
            "kizzasi_seek_zero_{}_{}.bin",
            std::process::id(),
            id
        ));

        let config = RecorderConfig {
            path: path.to_string_lossy().to_string(),
            format: RecorderFormat::Binary,
            sample_rate: 44100.0,
            channels: 1,
            buffer_size: 1024,
            record_timestamps: true,
            metadata: std::collections::HashMap::new(),
        };

        let mut recorder = StreamRecorder::new(config).await.unwrap();
        recorder.record_samples(&[1.0, 2.0], None).await.unwrap();
        recorder.record_samples(&[3.0, 4.0], None).await.unwrap();
        recorder.finalize().await.unwrap();

        let mut player = StreamPlayer::new(&path).await.unwrap();
        // Advance to frame 1
        let _ = player.next_frame().await.unwrap();
        // Seek back to 0
        player.seek_to_frame(0).await.unwrap();
        let frame = player.next_frame().await.unwrap().unwrap();
        assert_eq!(frame.frame_number, 0);
        assert_eq!(frame.samples, vec![1.0, 2.0]);

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn test_seek_past_end() {
        let temp_dir = std::env::temp_dir();
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(2);
        let path = temp_dir.join(format!(
            "kizzasi_seek_past_{}_{}.bin",
            std::process::id(),
            id
        ));

        let config = RecorderConfig {
            path: path.to_string_lossy().to_string(),
            format: RecorderFormat::Binary,
            sample_rate: 44100.0,
            channels: 1,
            buffer_size: 1024,
            record_timestamps: true,
            metadata: std::collections::HashMap::new(),
        };

        let mut recorder = StreamRecorder::new(config).await.unwrap();
        recorder.record_samples(&[1.0], None).await.unwrap();
        recorder.record_samples(&[2.0], None).await.unwrap();
        recorder.record_samples(&[3.0], None).await.unwrap();
        recorder.finalize().await.unwrap();

        let mut player = StreamPlayer::new(&path).await.unwrap();
        let result = player.seek_to_frame(10).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("3") || err_msg.contains("frame"),
            "Error message should mention frame count: {}",
            err_msg
        );

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn test_seek_no_timestamps() {
        let temp_dir = std::env::temp_dir();
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(3);
        let path = temp_dir.join(format!(
            "kizzasi_seek_nots_{}_{}.bin",
            std::process::id(),
            id
        ));

        let config = RecorderConfig {
            path: path.to_string_lossy().to_string(),
            format: RecorderFormat::Binary,
            sample_rate: 44100.0,
            channels: 1,
            buffer_size: 1024,
            record_timestamps: false,
            metadata: std::collections::HashMap::new(),
        };

        let mut recorder = StreamRecorder::new(config).await.unwrap();
        recorder.record_samples(&[10.0, 20.0], None).await.unwrap();
        recorder.record_samples(&[30.0, 40.0], None).await.unwrap();
        recorder.record_samples(&[50.0, 60.0], None).await.unwrap();
        recorder.finalize().await.unwrap();

        let mut player = StreamPlayer::new(&path).await.unwrap();
        player.seek_to_frame(2).await.unwrap();
        let frame = player.next_frame().await.unwrap().unwrap();
        assert_eq!(frame.frame_number, 2);
        assert_eq!(frame.samples, vec![50.0, 60.0]);

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn test_json_playback() {
        let temp_dir = std::env::temp_dir();
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(42);
        let path = temp_dir.join(format!("kizzasi_json_{}_{}.json", std::process::id(), id));

        let config = RecorderConfig {
            path: path.to_string_lossy().to_string(),
            format: RecorderFormat::Json,
            sample_rate: 44100.0,
            channels: 1,
            buffer_size: 1024,
            record_timestamps: true,
            metadata: std::collections::HashMap::new(),
        };

        let mut recorder = StreamRecorder::new(config).await.unwrap();
        recorder
            .record_samples(&[1.0, 2.0, 3.0], Some(0.1))
            .await
            .unwrap();
        recorder
            .record_samples(&[4.0, 5.0, 6.0], Some(0.2))
            .await
            .unwrap();
        recorder.finalize().await.unwrap();

        let mut player = StreamPlayer::new(&path).await.unwrap();

        let frame1 = player.next_frame().await.unwrap().unwrap();
        assert_eq!(frame1.samples, vec![1.0, 2.0, 3.0]);
        assert_eq!(frame1.frame_number, 0);
        assert!(frame1.timestamp.is_some());

        let frame2 = player.next_frame().await.unwrap().unwrap();
        assert_eq!(frame2.samples, vec![4.0, 5.0, 6.0]);
        assert_eq!(frame2.frame_number, 1);
        assert!(frame2.timestamp.is_some());

        assert!(player.next_frame().await.unwrap().is_none());

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn test_csv_playback() {
        let temp_dir = std::env::temp_dir();
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(43);
        let path = temp_dir.join(format!("kizzasi_csv_{}_{}.csv", std::process::id(), id));

        let config = RecorderConfig {
            path: path.to_string_lossy().to_string(),
            format: RecorderFormat::Csv,
            sample_rate: 44100.0,
            channels: 1,
            buffer_size: 1024,
            record_timestamps: true,
            metadata: std::collections::HashMap::new(),
        };

        let mut recorder = StreamRecorder::new(config).await.unwrap();
        recorder
            .record_samples(&[10.0, 20.0], Some(0.5))
            .await
            .unwrap();
        recorder
            .record_samples(&[30.0, 40.0], Some(1.0))
            .await
            .unwrap();
        recorder.finalize().await.unwrap();

        let mut player = StreamPlayer::new(&path).await.unwrap();

        let frame1 = player.next_frame().await.unwrap().unwrap();
        assert_eq!(frame1.samples, vec![10.0, 20.0]);
        assert_eq!(frame1.frame_number, 0);

        let frame2 = player.next_frame().await.unwrap().unwrap();
        assert_eq!(frame2.samples, vec![30.0, 40.0]);
        assert_eq!(frame2.frame_number, 1);

        assert!(player.next_frame().await.unwrap().is_none());

        let _ = tokio::fs::remove_file(&path).await;
    }

    // === Regression tests: binary format desync when an explicit timestamp
    // is passed with record_timestamps = false (high) ===
    //
    // The binary frame layout carries no per-frame "has timestamp" flag;
    // the reader decides whether to consume 8 timestamp bytes purely from
    // the header's `record_timestamps` config. `record_samples` must
    // therefore never write a timestamp for a recording configured with
    // `record_timestamps = false`, even when the caller passes one
    // explicitly -- otherwise the extra 8 bytes get interpreted as the next
    // frame's sample count, corrupting every subsequent frame.

    #[tokio::test]
    async fn test_explicit_timestamp_ignored_when_record_timestamps_false() {
        let temp_dir = std::env::temp_dir();
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(4);
        let path = temp_dir.join(format!(
            "kizzasi_ts_desync_{}_{}.bin",
            std::process::id(),
            id
        ));

        let config = RecorderConfig {
            path: path.to_string_lossy().to_string(),
            format: RecorderFormat::Binary,
            sample_rate: 44100.0,
            channels: 1,
            buffer_size: 1024,
            record_timestamps: false,
            metadata: std::collections::HashMap::new(),
        };

        let mut recorder = StreamRecorder::new(config).await.unwrap();
        // Explicit timestamps despite record_timestamps = false: before the
        // fix, write_frame would still emit 8 timestamp bytes per frame
        // that the reader never expected to skip.
        recorder
            .record_samples(&[1.0, 2.0, 3.0], Some(0.111))
            .await
            .unwrap();
        recorder
            .record_samples(&[4.0, 5.0, 6.0], Some(0.222))
            .await
            .unwrap();
        recorder
            .record_samples(&[7.0, 8.0, 9.0], Some(0.333))
            .await
            .unwrap();
        recorder.finalize().await.unwrap();

        let mut player = StreamPlayer::new(&path).await.unwrap();

        let frame1 = player.next_frame().await.unwrap().unwrap();
        assert_eq!(frame1.samples, vec![1.0, 2.0, 3.0]);
        assert_eq!(frame1.timestamp, None);

        let frame2 = player.next_frame().await.unwrap().unwrap();
        assert_eq!(
            frame2.samples,
            vec![4.0, 5.0, 6.0],
            "frame 2 must not be corrupted by misread timestamp bytes from frame 1"
        );
        assert_eq!(frame2.timestamp, None);

        let frame3 = player.next_frame().await.unwrap().unwrap();
        assert_eq!(frame3.samples, vec![7.0, 8.0, 9.0]);
        assert_eq!(frame3.timestamp, None);

        assert!(player.next_frame().await.unwrap().is_none());

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn test_explicit_timestamp_preserved_when_record_timestamps_true() {
        let temp_dir = std::env::temp_dir();
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(5);
        let path = temp_dir.join(format!(
            "kizzasi_ts_explicit_{}_{}.bin",
            std::process::id(),
            id
        ));

        let config = RecorderConfig {
            path: path.to_string_lossy().to_string(),
            format: RecorderFormat::Binary,
            sample_rate: 44100.0,
            channels: 1,
            buffer_size: 1024,
            record_timestamps: true,
            metadata: std::collections::HashMap::new(),
        };

        let mut recorder = StreamRecorder::new(config).await.unwrap();
        recorder
            .record_samples(&[1.0, 2.0], Some(12.5))
            .await
            .unwrap();
        recorder.finalize().await.unwrap();

        let mut player = StreamPlayer::new(&path).await.unwrap();
        let frame = player.next_frame().await.unwrap().unwrap();
        assert_eq!(frame.samples, vec![1.0, 2.0]);
        assert_eq!(frame.timestamp, Some(12.5));

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn test_default_timestamp_computed_when_record_timestamps_true_and_none_passed() {
        let temp_dir = std::env::temp_dir();
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(6);
        let path = temp_dir.join(format!(
            "kizzasi_ts_default_{}_{}.bin",
            std::process::id(),
            id
        ));

        let config = RecorderConfig {
            path: path.to_string_lossy().to_string(),
            format: RecorderFormat::Binary,
            sample_rate: 44100.0,
            channels: 1,
            buffer_size: 1024,
            record_timestamps: true,
            metadata: std::collections::HashMap::new(),
        };

        let mut recorder = StreamRecorder::new(config).await.unwrap();
        recorder.record_samples(&[1.0], None).await.unwrap();
        recorder.finalize().await.unwrap();

        let mut player = StreamPlayer::new(&path).await.unwrap();
        let frame = player.next_frame().await.unwrap().unwrap();
        assert_eq!(frame.samples, vec![1.0]);
        assert!(
            frame.timestamp.is_some(),
            "record_timestamps = true must always yield a timestamp, even without an explicit one"
        );

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn test_seek_stays_in_sync_with_explicit_timestamps_and_no_record_timestamps() {
        let temp_dir = std::env::temp_dir();
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(7);
        let path = temp_dir.join(format!("kizzasi_ts_seek_{}_{}.bin", std::process::id(), id));

        let config = RecorderConfig {
            path: path.to_string_lossy().to_string(),
            format: RecorderFormat::Binary,
            sample_rate: 44100.0,
            channels: 1,
            buffer_size: 1024,
            record_timestamps: false,
            metadata: std::collections::HashMap::new(),
        };

        let mut recorder = StreamRecorder::new(config).await.unwrap();
        for i in 0usize..5 {
            let v = i as f32;
            // Every call passes an explicit timestamp even though the
            // recording is configured with record_timestamps = false.
            recorder
                .record_samples(&[v, v + 1.0], Some(v as f64))
                .await
                .unwrap();
        }
        recorder.finalize().await.unwrap();

        let mut player = StreamPlayer::new(&path).await.unwrap();
        player.seek_to_frame(3).await.unwrap();
        let frame = player.next_frame().await.unwrap().unwrap();
        assert_eq!(frame.frame_number, 3);
        assert_eq!(frame.samples, vec![3.0, 4.0]);
        assert_eq!(frame.timestamp, None);

        let _ = tokio::fs::remove_file(&path).await;
    }
}
