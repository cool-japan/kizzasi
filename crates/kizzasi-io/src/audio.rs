//! Audio input/output via cpal
//!
//! Provides comprehensive audio I/O with:
//! - Audio input (recording)
//! - Audio output (playback)
//! - Multi-channel support
//! - WAV file integration
//! - Device enumeration
//! - ASIO backend support (Windows)
//! - JACK backend support (Linux/macOS)

use crate::error::{IoError, IoResult};
use crate::stream::{SignalStream, StreamConfig};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

#[cfg(feature = "file")]
use crate::file::WavReader;

/// Audio backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AudioBackend {
    /// System default backend (WASAPI on Windows, CoreAudio on macOS, ALSA on Linux)
    #[default]
    Default,
    /// ASIO backend (Windows only, requires ASIO driver installation)
    #[cfg(target_os = "windows")]
    Asio,
    /// JACK Audio Connection Kit (Linux/macOS pro audio)
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    Jack,
}

/// Configuration for audio I/O
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Device name (None for default)
    pub device_name: Option<String>,

    /// Sample rate
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,

    /// Number of channels
    #[serde(default = "default_channels")]
    pub channels: u16,

    /// Buffer size
    #[serde(default = "default_buffer_size")]
    pub buffer_size: u32,

    /// Use output device (false for input)
    #[serde(default)]
    pub output: bool,

    /// Audio backend to use
    #[serde(default)]
    pub backend: AudioBackend,
}

fn default_sample_rate() -> u32 {
    44100
}

fn default_channels() -> u16 {
    1
}

fn default_buffer_size() -> u32 {
    1024
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device_name: None,
            sample_rate: 44100,
            channels: 1,
            buffer_size: 1024,
            output: false,
            backend: AudioBackend::Default,
        }
    }
}

impl AudioConfig {
    /// Create new audio configuration for input
    pub fn new() -> Self {
        Self::default()
    }

    /// Create new audio configuration for output
    pub fn new_output() -> Self {
        Self {
            output: true,
            ..Default::default()
        }
    }

    /// Set sample rate
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Set number of channels
    pub fn channels(mut self, n: u16) -> Self {
        self.channels = n;
        self
    }

    /// Set buffer size
    pub fn buffer_size(mut self, size: u32) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set device name
    pub fn device(mut self, name: &str) -> Self {
        self.device_name = Some(name.to_string());
        self
    }

    /// Set audio backend
    pub fn backend(mut self, backend: AudioBackend) -> Self {
        self.backend = backend;
        self
    }
}

/// Get the appropriate audio host based on backend selection
fn get_host(backend: AudioBackend) -> IoResult<cpal::Host> {
    match backend {
        AudioBackend::Default => Ok(cpal::default_host()),
        #[cfg(target_os = "windows")]
        AudioBackend::Asio => {
            // ASIO backend - iterate through available hosts
            let available_hosts = cpal::available_hosts();
            if available_hosts.contains(&cpal::HostId::Asio) {
                Ok(cpal::host_from_id(cpal::HostId::Asio))
            } else {
                Err(IoError::ConfigError(
                    "ASIO backend not available. Ensure ASIO drivers are installed.".into(),
                ))
            }
        }
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        AudioBackend::Jack => {
            // JACK backend - for now, fall back to default as JACK support varies by cpal version
            // In cpal 0.16, JACK may not be directly available via HostId
            // Users can use JACK-enabled systems with the default host
            info!("JACK support requested - using default host (ensure JACK is configured as default)");
            Ok(cpal::default_host())
        }
    }
}

/// Audio input stream
pub struct AudioInput {
    config: StreamConfig,
    audio_config: AudioConfig,
    buffer: Arc<Mutex<Vec<f32>>>,
    multi_channel_buffer: Arc<Mutex<Vec<Vec<f32>>>>,
    #[allow(dead_code)]
    stream: Option<cpal::Stream>,
    active: bool,
}

impl AudioInput {
    /// Create a new audio input
    pub fn new(audio_config: AudioConfig) -> IoResult<Self> {
        let stream_config = StreamConfig {
            sample_rate: audio_config.sample_rate as f32,
            channels: audio_config.channels as usize,
            buffer_size: audio_config.buffer_size as usize,
            timeout: None,
        };

        let num_channels = audio_config.channels as usize;

        Ok(Self {
            config: stream_config,
            audio_config,
            buffer: Arc::new(Mutex::new(Vec::new())),
            multi_channel_buffer: Arc::new(Mutex::new(vec![Vec::new(); num_channels])),
            stream: None,
            active: false,
        })
    }

    /// Start capturing audio
    pub fn start(&mut self) -> IoResult<()> {
        let host = get_host(self.audio_config.backend)?;

        let device = if let Some(ref name) = self.audio_config.device_name {
            host.input_devices()
                .map_err(|e| IoError::ConfigError(e.to_string()))?
                .find(|d| d.name().map(|n| n == *name).unwrap_or(false))
                .ok_or_else(|| IoError::ConfigError(format!("Device not found: {}", name)))?
        } else {
            host.default_input_device()
                .ok_or_else(|| IoError::ConfigError("No default input device".into()))?
        };

        let config = cpal::StreamConfig {
            channels: self.audio_config.channels,
            sample_rate: cpal::SampleRate(self.audio_config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.audio_config.buffer_size),
        };

        info!(
            "Starting audio input: {}Hz, {} channels, buffer={}",
            self.audio_config.sample_rate,
            self.audio_config.channels,
            self.audio_config.buffer_size
        );

        let buffer = self.buffer.clone();
        let multi_buffer = self.multi_channel_buffer.clone();
        let num_channels = self.audio_config.channels as usize;

        let stream = device
            .build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    // Store interleaved samples
                    if let Ok(mut buf) = buffer.lock() {
                        buf.extend_from_slice(data);
                    }

                    // Store de-interleaved multi-channel samples
                    if let Ok(mut multi_buf) = multi_buffer.lock() {
                        for (i, &sample) in data.iter().enumerate() {
                            let channel = i % num_channels;
                            multi_buf[channel].push(sample);
                        }
                    }
                },
                |err| {
                    warn!("Audio stream error: {}", err);
                },
                None,
            )
            .map_err(|e| IoError::StreamError(e.to_string()))?;

        stream
            .play()
            .map_err(|e| IoError::StreamError(e.to_string()))?;

        self.stream = Some(stream);
        self.active = true;

        info!("Audio input started");
        Ok(())
    }

    /// Stop capturing audio
    pub fn stop(&mut self) -> IoResult<()> {
        self.stream = None;
        self.active = false;
        info!("Audio input stopped");
        Ok(())
    }

    /// Read multi-channel data
    pub fn read_channels(&mut self) -> IoResult<Array2<f32>> {
        let mut multi_buffer = self
            .multi_channel_buffer
            .lock()
            .map_err(|_| IoError::StreamError("Buffer lock failed".into()))?;

        // Find minimum length across all channels
        let min_len = multi_buffer
            .iter()
            .map(|ch| ch.len())
            .min()
            .unwrap_or(0)
            .min(self.config.buffer_size);

        if min_len == 0 {
            return Ok(Array2::zeros((
                self.config.buffer_size,
                self.config.channels,
            )));
        }

        let mut result = Array2::zeros((min_len, self.config.channels));
        for (ch_idx, channel_data) in multi_buffer.iter_mut().enumerate() {
            let samples: Vec<f32> = channel_data.drain(..min_len).collect();
            for (i, &sample) in samples.iter().enumerate() {
                result[[i, ch_idx]] = sample;
            }
        }

        debug!(
            "Read {} frames from {} channels",
            min_len, self.config.channels
        );
        Ok(result)
    }

    /// Get available input devices for a specific backend
    pub fn list_devices_with_backend(backend: AudioBackend) -> IoResult<Vec<String>> {
        let host = get_host(backend)?;
        let devices = host
            .input_devices()
            .map_err(|e| IoError::ConfigError(e.to_string()))?;

        let names: Vec<String> = devices.filter_map(|d| d.name().ok()).collect();
        Ok(names)
    }

    /// Get available input devices (using default backend)
    pub fn list_devices() -> IoResult<Vec<String>> {
        Self::list_devices_with_backend(AudioBackend::Default)
    }
}

impl SignalStream for AudioInput {
    fn read(&mut self) -> IoResult<Array1<f32>> {
        let mut buffer = self
            .buffer
            .lock()
            .map_err(|_| IoError::StreamError("Buffer lock failed".into()))?;

        let size = self.config.buffer_size.min(buffer.len());
        if size == 0 {
            return Ok(Array1::zeros(self.config.buffer_size));
        }

        let data: Vec<f32> = buffer.drain(..size).collect();
        let mut result = Array1::zeros(self.config.buffer_size);
        for (i, val) in data.into_iter().enumerate() {
            result[i] = val;
        }
        Ok(result)
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn config(&self) -> &StreamConfig {
        &self.config
    }

    fn close(&mut self) -> IoResult<()> {
        self.stop()
    }
}

/// Audio output stream (playback)
pub struct AudioOutput {
    #[allow(dead_code)]
    config: StreamConfig,
    audio_config: AudioConfig,
    buffer: Arc<Mutex<Vec<f32>>>,
    #[allow(dead_code)]
    stream: Option<cpal::Stream>,
    active: bool,
    underrun_count: Arc<Mutex<usize>>,
}

impl AudioOutput {
    /// Create a new audio output
    pub fn new(audio_config: AudioConfig) -> IoResult<Self> {
        let stream_config = StreamConfig {
            sample_rate: audio_config.sample_rate as f32,
            channels: audio_config.channels as usize,
            buffer_size: audio_config.buffer_size as usize,
            timeout: None,
        };

        Ok(Self {
            config: stream_config,
            audio_config,
            buffer: Arc::new(Mutex::new(Vec::new())),
            stream: None,
            active: false,
            underrun_count: Arc::new(Mutex::new(0)),
        })
    }

    /// Start audio playback
    pub fn start(&mut self) -> IoResult<()> {
        let host = get_host(self.audio_config.backend)?;

        let device = if let Some(ref name) = self.audio_config.device_name {
            host.output_devices()
                .map_err(|e| IoError::ConfigError(e.to_string()))?
                .find(|d| d.name().map(|n| n == *name).unwrap_or(false))
                .ok_or_else(|| IoError::ConfigError(format!("Device not found: {}", name)))?
        } else {
            host.default_output_device()
                .ok_or_else(|| IoError::ConfigError("No default output device".into()))?
        };

        let config = cpal::StreamConfig {
            channels: self.audio_config.channels,
            sample_rate: cpal::SampleRate(self.audio_config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.audio_config.buffer_size),
        };

        info!(
            "Starting audio output: {}Hz, {} channels, buffer={}",
            self.audio_config.sample_rate,
            self.audio_config.channels,
            self.audio_config.buffer_size
        );

        let buffer = self.buffer.clone();
        let underrun_count = self.underrun_count.clone();

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let mut buf = match buffer.lock() {
                        Ok(b) => b,
                        Err(_) => return,
                    };

                    if buf.len() >= data.len() {
                        // Sufficient data available
                        for sample in data.iter_mut() {
                            *sample = buf.remove(0);
                        }
                    } else {
                        // Buffer underrun - fill with zeros
                        if let Ok(mut count) = underrun_count.lock() {
                            *count += 1;
                        }
                        for sample in data.iter_mut() {
                            *sample = 0.0;
                        }
                    }
                },
                |err| {
                    warn!("Audio output error: {}", err);
                },
                None,
            )
            .map_err(|e| IoError::StreamError(e.to_string()))?;

        stream
            .play()
            .map_err(|e| IoError::StreamError(e.to_string()))?;

        self.stream = Some(stream);
        self.active = true;

        info!("Audio output started");
        Ok(())
    }

    /// Stop audio playback
    pub fn stop(&mut self) -> IoResult<()> {
        self.stream = None;
        self.active = false;
        info!("Audio output stopped");
        Ok(())
    }

    /// Write samples to playback buffer
    pub fn write(&mut self, samples: &Array1<f32>) -> IoResult<()> {
        let mut buffer = self
            .buffer
            .lock()
            .map_err(|_| IoError::StreamError("Buffer lock failed".into()))?;

        buffer.extend(samples.iter());
        debug!("Wrote {} samples to output buffer", samples.len());
        Ok(())
    }

    /// Write multi-channel samples (interleaved)
    pub fn write_channels(&mut self, samples: &Array2<f32>) -> IoResult<()> {
        let mut buffer = self
            .buffer
            .lock()
            .map_err(|_| IoError::StreamError("Buffer lock failed".into()))?;

        // Interleave channels
        for row in samples.outer_iter() {
            buffer.extend(row.iter());
        }

        debug!(
            "Wrote {} frames from {} channels to output buffer",
            samples.nrows(),
            samples.ncols()
        );
        Ok(())
    }

    /// Get buffer level (number of samples queued)
    pub fn buffer_level(&self) -> usize {
        self.buffer.lock().map(|b| b.len()).unwrap_or(0)
    }

    /// Get underrun count
    pub fn underrun_count(&self) -> usize {
        self.underrun_count.lock().map(|c| *c).unwrap_or(0)
    }

    /// Clear buffer
    pub fn clear_buffer(&mut self) -> IoResult<()> {
        let mut buffer = self
            .buffer
            .lock()
            .map_err(|_| IoError::StreamError("Buffer lock failed".into()))?;

        buffer.clear();
        Ok(())
    }

    /// Get available output devices for a specific backend
    pub fn list_devices_with_backend(backend: AudioBackend) -> IoResult<Vec<String>> {
        let host = get_host(backend)?;
        let devices = host
            .output_devices()
            .map_err(|e| IoError::ConfigError(e.to_string()))?;

        let names: Vec<String> = devices.filter_map(|d| d.name().ok()).collect();
        Ok(names)
    }

    /// Get available output devices (using default backend)
    pub fn list_devices() -> IoResult<Vec<String>> {
        Self::list_devices_with_backend(AudioBackend::Default)
    }

    /// Play a WAV file
    #[cfg(feature = "file")]
    pub async fn play_wav_file(&mut self, path: &str) -> IoResult<()> {
        let reader = WavReader::open(path).await?;
        let spec = reader.spec();

        // Check compatibility
        if spec.sample_rate != self.audio_config.sample_rate {
            warn!(
                "WAV sample rate ({}) differs from output config ({})",
                spec.sample_rate, self.audio_config.sample_rate
            );
        }

        if spec.channels != self.audio_config.channels {
            return Err(IoError::ConfigError(format!(
                "WAV channels ({}) differ from output config ({})",
                spec.channels, self.audio_config.channels
            )));
        }

        // Load and write samples
        let samples = reader.read_all().await?;
        self.write(&samples)?;

        info!("Loaded WAV file: {} samples", samples.len());
        Ok(())
    }

    /// Check if active
    pub fn is_active(&self) -> bool {
        self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_config() {
        let config = AudioConfig::new()
            .sample_rate(48000)
            .channels(2)
            .buffer_size(2048);

        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.channels, 2);
        assert_eq!(config.buffer_size, 2048);
        assert!(!config.output);
    }

    #[test]
    fn test_audio_config_output() {
        let config = AudioConfig::new_output().sample_rate(44100).channels(1);

        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channels, 1);
        assert!(config.output);
    }

    #[test]
    fn test_list_input_devices() {
        let result = AudioInput::list_devices();
        assert!(result.is_ok());
    }

    #[test]
    fn test_list_output_devices() {
        let result = AudioOutput::list_devices();
        assert!(result.is_ok());
    }
}
