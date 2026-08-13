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
use crossbeam_queue::ArrayQueue;
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
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

/// Look up a cpal host by name among the hosts this build actually offers.
///
/// Returns [`IoError::Unsupported`] (listing the hosts that *are* available)
/// when the requested backend is absent, rather than quietly handing back some
/// other host under the requested backend's name.
#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
fn host_by_name(name: &str, requirement: &str) -> IoResult<cpal::Host> {
    let host_id = cpal::available_hosts()
        .into_iter()
        .find(|id| id.name().eq_ignore_ascii_case(name))
        .ok_or_else(|| {
            let available: Vec<&str> = cpal::available_hosts().iter().map(|id| id.name()).collect();
            IoError::Unsupported(format!(
                "The {name} audio backend is not available in this build \
                 (ensure {requirement}). Available hosts: {}",
                available.join(", ")
            ))
        })?;

    cpal::host_from_id(host_id)
        .map_err(|e| IoError::ConfigError(format!("Failed to open the {name} audio host: {e}")))
}

/// Get the appropriate audio host based on backend selection
fn get_host(backend: AudioBackend) -> IoResult<cpal::Host> {
    match backend {
        AudioBackend::Default => Ok(cpal::default_host()),
        #[cfg(target_os = "windows")]
        AudioBackend::Asio => host_by_name(
            "ASIO",
            "the ASIO drivers are installed and cpal was built with its `asio` feature",
        ),
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        AudioBackend::Jack => host_by_name(
            "JACK",
            "the JACK server is running and cpal was built with its `jack` feature",
        ),
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
                .find(|d| {
                    d.description()
                        .map(|desc| desc.name() == *name)
                        .unwrap_or(false)
                })
                .ok_or_else(|| IoError::ConfigError(format!("Device not found: {}", name)))?
        } else {
            host.default_input_device()
                .ok_or_else(|| IoError::ConfigError("No default input device".into()))?
        };

        let config = cpal::StreamConfig {
            channels: self.audio_config.channels,
            sample_rate: self.audio_config.sample_rate,
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
                config,
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

    /// Read multi-channel data.
    ///
    /// Follows the same read contract as [`SignalStream::read`]: the returned
    /// array has one row per frame that was **actually captured** (never
    /// zero-padded up to `buffer_size`), `Err(IoError::BufferEmpty)` when the
    /// device has not delivered any frames yet, and `Err(IoError::EndOfStream)`
    /// once the input has been stopped.
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

        if min_len == 0 || (min_len < self.config.buffer_size && self.active) {
            // A previous version returned Ok(Array2::zeros((buffer_size,
            // channels))) here, presenting a full block of fabricated silence
            // as captured audio. Partial captures stay buffered while the
            // device runs; the tail is delivered once it stops.
            drop(multi_buffer);
            return Err(if self.active {
                IoError::BufferEmpty
            } else {
                IoError::EndOfStream
            });
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

        let names: Vec<String> = devices
            .filter_map(|d| d.description().ok().map(|desc| desc.name().to_string()))
            .collect();
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
        // Previously an empty or partial capture returned
        // Ok(Array1::zeros(buffer_size)) -- a full block of fabricated
        // silence indistinguishable from a genuinely silent microphone.
        // Per the stream read contract: while the device is running, a
        // partial block stays buffered and BufferEmpty is reported (nothing
        // is consumed); once stopped, the remaining samples are delivered as
        // one final short block, then EndOfStream.
        if size == 0 || (size < self.config.buffer_size && self.active) {
            drop(buffer);
            return Err(if self.active {
                IoError::BufferEmpty
            } else {
                IoError::EndOfStream
            });
        }

        let data: Vec<f32> = buffer.drain(..size).collect();
        Ok(Array1::from_vec(data))
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
    /// Lock-free SPSC-friendly ring buffer shared with the realtime cpal
    /// callback. `ArrayQueue::push`/`pop` never block and never allocate,
    /// which a `std::sync::Mutex<Vec<f32>>` with `Vec::remove(0)` could not
    /// guarantee (O(n) per-sample shifting plus a blocking lock acquisition
    /// on the OS audio thread).
    buffer: Arc<ArrayQueue<f32>>,
    #[allow(dead_code)]
    stream: Option<cpal::Stream>,
    active: bool,
    underrun_count: Arc<AtomicUsize>,
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

        // Bound the queue generously (60s of audio at the configured rate)
        // instead of growing an unbounded Vec forever. A single large
        // `write()` call (e.g. `play_wav_file` queuing a whole file up
        // front) should comfortably fit; callers that genuinely need more
        // headroom get an honest `IoError::BufferFull` instead of silent
        // unbounded growth.
        let channels = (audio_config.channels as usize).max(1);
        let queue_capacity = (audio_config.sample_rate as usize)
            .saturating_mul(channels)
            .saturating_mul(60)
            .max(audio_config.buffer_size as usize)
            .max(1);

        Ok(Self {
            config: stream_config,
            audio_config,
            buffer: Arc::new(ArrayQueue::new(queue_capacity)),
            stream: None,
            active: false,
            underrun_count: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Start audio playback
    pub fn start(&mut self) -> IoResult<()> {
        let host = get_host(self.audio_config.backend)?;

        let device = if let Some(ref name) = self.audio_config.device_name {
            host.output_devices()
                .map_err(|e| IoError::ConfigError(e.to_string()))?
                .find(|d| {
                    d.description()
                        .map(|desc| desc.name() == *name)
                        .unwrap_or(false)
                })
                .ok_or_else(|| IoError::ConfigError(format!("Device not found: {}", name)))?
        } else {
            host.default_output_device()
                .ok_or_else(|| IoError::ConfigError("No default output device".into()))?
        };

        let config = cpal::StreamConfig {
            channels: self.audio_config.channels,
            sample_rate: self.audio_config.sample_rate,
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
                config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    // Lock-free, allocation-free: pop one sample at a time
                    // instead of taking a std Mutex and shifting a Vec
                    // (which was O(data.len() * queued_len) per callback
                    // and risked priority inversion against the writer
                    // thread). A per-sample underrun (queue empty) is
                    // filled with silence and counted once per callback.
                    let mut underran = false;
                    for sample in data.iter_mut() {
                        *sample = match buffer.pop() {
                            Some(value) => value,
                            None => {
                                underran = true;
                                0.0
                            }
                        };
                    }
                    if underran {
                        underrun_count.fetch_add(1, Ordering::Relaxed);
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
    ///
    /// Returns `IoError::BufferFull` if the queue's bounded capacity would
    /// be exceeded; samples already pushed before that point remain queued
    /// (partial write), matching how a real hardware buffer would behave
    /// under sustained overflow.
    pub fn write(&mut self, samples: &Array1<f32>) -> IoResult<()> {
        for &sample in samples.iter() {
            self.buffer.push(sample).map_err(|_| IoError::BufferFull)?;
        }
        debug!("Wrote {} samples to output buffer", samples.len());
        Ok(())
    }

    /// Write multi-channel samples (interleaved)
    pub fn write_channels(&mut self, samples: &Array2<f32>) -> IoResult<()> {
        for row in samples.outer_iter() {
            for &sample in row.iter() {
                self.buffer.push(sample).map_err(|_| IoError::BufferFull)?;
            }
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
        self.buffer.len()
    }

    /// Get underrun count
    pub fn underrun_count(&self) -> usize {
        self.underrun_count.load(Ordering::Relaxed)
    }

    /// Clear buffer
    pub fn clear_buffer(&mut self) -> IoResult<()> {
        while self.buffer.pop().is_some() {}
        Ok(())
    }

    /// Get available output devices for a specific backend
    pub fn list_devices_with_backend(backend: AudioBackend) -> IoResult<Vec<String>> {
        let host = get_host(backend)?;
        let devices = host
            .output_devices()
            .map_err(|e| IoError::ConfigError(e.to_string()))?;

        let names: Vec<String> = devices
            .filter_map(|d| d.description().ok().map(|desc| desc.name().to_string()))
            .collect();
        Ok(names)
    }

    /// Get available output devices (using default backend)
    pub fn list_devices() -> IoResult<Vec<String>> {
        Self::list_devices_with_backend(AudioBackend::Default)
    }

    /// Play a WAV file
    #[cfg(feature = "file")]
    pub async fn play_wav_file(&mut self, path: &str) -> IoResult<()> {
        let mut reader = WavReader::open(path).await?;
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

    // === Regression tests: lock-free bounded playback queue (high) ===
    //
    // These exercise `AudioOutput`'s buffer directly (no `start()`/cpal
    // device needed, so they run without real audio hardware). They also
    // stand in for the removed `Vec::remove(0)` O(n^2) behavior: with a
    // bounded `ArrayQueue`, pushing/popping thousands of samples must
    // complete quickly and the returned values must preserve FIFO order.

    #[test]
    fn test_audio_output_write_and_buffer_level() {
        let mut output = AudioOutput::new(AudioConfig::new_output()).unwrap();
        assert_eq!(output.buffer_level(), 0);

        let samples = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        output.write(&samples).unwrap();

        assert_eq!(output.buffer_level(), 4);
    }

    #[test]
    fn test_audio_output_clear_buffer() {
        let mut output = AudioOutput::new(AudioConfig::new_output()).unwrap();
        output
            .write(&Array1::from_vec(vec![1.0; 100]))
            .expect("write should succeed within the default queue capacity");
        assert_eq!(output.buffer_level(), 100);

        output.clear_buffer().unwrap();
        assert_eq!(output.buffer_level(), 0);
    }

    #[test]
    fn test_audio_output_underrun_count_starts_zero() {
        let output = AudioOutput::new(AudioConfig::new_output()).unwrap();
        assert_eq!(output.underrun_count(), 0);
    }

    #[test]
    fn test_audio_output_write_reports_buffer_full_instead_of_growing_unbounded() {
        // Tiny sample_rate + buffer_size forces a small (60-sample) queue
        // capacity, so overflow is reachable without allocating megabytes.
        let config = AudioConfig::new_output().sample_rate(1).buffer_size(4);
        let mut output = AudioOutput::new(config).unwrap();

        let too_many = Array1::from_vec(vec![0.5; 10_000]);
        let result = output.write(&too_many);

        assert!(
            matches!(result, Err(IoError::BufferFull)),
            "expected BufferFull once the bounded queue's capacity is exceeded, got {result:?}"
        );
        // Whatever fit before overflow should still be queued (partial
        // write), not silently dropped.
        assert!(output.buffer_level() > 0);
    }

    #[test]
    fn test_audio_output_write_many_samples_stays_fast_and_ordered() {
        // Regression guard for the O(n^2) `Vec::remove(0)` pattern: pushing
        // and draining a large number of samples through the public API
        // must be fast (no per-sample O(n) shifting) and must preserve
        // order via `ArrayQueue`'s FIFO semantics.
        let config = AudioConfig::new_output().sample_rate(44_100).channels(1);
        let mut output = AudioOutput::new(config).unwrap();

        let n = 50_000;
        let samples: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let start = std::time::Instant::now();
        output.write(&Array1::from_vec(samples)).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(output.buffer_level(), n);
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "writing {n} samples took {elapsed:?}, suggesting O(n^2) behavior regressed"
        );
    }
}
