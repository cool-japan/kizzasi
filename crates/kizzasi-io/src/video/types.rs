//! Video value types: frames, pixel formats, sources, reader configuration,
//! and the camera/metadata descriptors used to open and describe a stream.
//!
//! This is a split of the former `video.rs`; see the parent `video` module
//! for the module-level documentation and honesty notes about what the
//! FFmpeg-backed reader actually supports.

use crate::error::{IoError, IoResult};
use scirs2_core::ndarray::{Array3, ArrayView3};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Video frame data
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Frame index in the video stream
    pub index: u64,

    /// Timestamp in seconds
    pub timestamp: f64,

    /// Frame width in pixels
    pub width: usize,

    /// Frame height in pixels
    pub height: usize,

    /// Number of color channels (1 for grayscale, 3 for RGB, 4 for RGBA)
    pub channels: usize,

    /// Raw frame data in row-major order (height x width x channels)
    /// For RGB: channels are in order R, G, B
    /// For grayscale: single channel
    pub data: Vec<u8>,
}

impl VideoFrame {
    /// Validate that `data` really holds `height * width * channels` bytes.
    ///
    /// All fields of [`VideoFrame`] are public, so a caller (or a decoder
    /// bug) can produce an inconsistent frame. Every operation on the frame
    /// validates first and returns an error rather than panicking -- the
    /// previous code called `.expect("Invalid frame dimensions")` and indexed
    /// `data[idx + 2]` unconditionally, so a mismatched frame aborted the
    /// process.
    pub fn validate(&self) -> IoResult<()> {
        if self.channels == 0 {
            return Err(IoError::ConfigError(
                "VideoFrame must have at least one channel".into(),
            ));
        }
        let expected = self
            .height
            .checked_mul(self.width)
            .and_then(|pixels| pixels.checked_mul(self.channels))
            .ok_or_else(|| {
                IoError::ConfigError("VideoFrame dimensions overflow usize".to_string())
            })?;
        if self.data.len() != expected {
            return Err(IoError::ConfigError(format!(
                "VideoFrame data length {} does not match {}x{}x{} = {}",
                self.data.len(),
                self.height,
                self.width,
                self.channels,
                expected
            )));
        }
        Ok(())
    }

    /// Convert frame to ndarray (height x width x channels)
    pub fn to_array(&self) -> IoResult<Array3<u8>> {
        self.validate()?;
        Array3::from_shape_vec((self.height, self.width, self.channels), self.data.clone())
            .map_err(|e| IoError::ConfigError(format!("Invalid frame dimensions: {e}")))
    }

    /// Get frame data as array view
    pub fn as_array(&self) -> IoResult<ArrayView3<'_, u8>> {
        self.validate()?;
        ArrayView3::from_shape((self.height, self.width, self.channels), &self.data)
            .map_err(|e| IoError::ConfigError(format!("Invalid frame dimensions: {e}")))
    }

    /// Convert to grayscale (if not already).
    ///
    /// Uses the ITU-R BT.601 luma coefficients over the first three channels.
    /// Frames with 2 channels (or any layout with no RGB triple) are rejected
    /// with [`IoError::Unsupported`] instead of reading past the end of a row.
    pub fn to_grayscale(&self) -> IoResult<VideoFrame> {
        self.validate()?;

        if self.channels == 1 {
            return Ok(self.clone());
        }
        if self.channels < 3 {
            return Err(IoError::Unsupported(format!(
                "Cannot convert a {}-channel frame to grayscale (need 1, or >= 3 with an RGB triple)",
                self.channels
            )));
        }

        let mut gray_data = Vec::with_capacity(self.width * self.height);

        for pixel in self.data.chunks_exact(self.channels) {
            let r = f32::from(pixel[0]);
            let g = f32::from(pixel[1]);
            let b = f32::from(pixel[2]);

            // ITU-R BT.601 luma coefficients
            let gray = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0) as u8;
            gray_data.push(gray);
        }

        Ok(VideoFrame {
            index: self.index,
            timestamp: self.timestamp,
            width: self.width,
            height: self.height,
            channels: 1,
            data: gray_data,
        })
    }

    /// Convert to f32 normalized to [0.0, 1.0]
    pub fn to_normalized_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&x| f32::from(x) / 255.0).collect()
    }
}

/// Video pixel format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelFormat {
    /// RGB 24-bit (8 bits per channel)
    Rgb,
    /// Grayscale 8-bit
    Gray,
    /// RGBA 32-bit (8 bits per channel)
    Rgba,
}

impl PixelFormat {
    /// Get number of channels
    pub fn channels(&self) -> usize {
        match self {
            PixelFormat::Gray => 1,
            PixelFormat::Rgb => 3,
            PixelFormat::Rgba => 4,
        }
    }
}

/// Video source type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoSource {
    /// File path
    File(PathBuf),
    /// Camera device (index or name)
    Camera(String),
    /// Network stream (RTSP, HTTP, etc.)
    Network(String),
}

/// Which decode/capture implementation `VideoReader` should use.
///
/// See the `video` module's own documentation for the exact resolution
/// rules and for the current division of labour between the two backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VideoBackend {
    /// Pick a backend per source: prefer the pure-Rust backend where it is
    /// capable of the source, fall back to FFmpeg otherwise. Errors only if
    /// neither backend can handle the source. In this release that means
    /// Y4M files and cameras go to the pure backend (when `video-pure` is
    /// compiled in, and for cameras only on a platform whose capture API
    /// `oximedia-capture` implements) and everything else to FFmpeg.
    #[default]
    Auto,
    /// Force the pure-Rust backend (the OxiMedia stack, feature
    /// `video-pure`). It decodes Y4M (YUV4MPEG2) files and captures from
    /// cameras; anything else -- another container, a network stream -- is an
    /// [`IoError::Unsupported`] naming what is missing, never a silent
    /// fallback to FFmpeg. Also errors if `video-pure` is not compiled in.
    Pure,
    /// Force the FFmpeg backend. Errors if the `video` feature is not
    /// compiled in.
    Ffmpeg,
}

/// Video reader configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoConfig {
    /// Video source
    pub source: VideoSource,

    /// Output pixel format
    pub pixel_format: PixelFormat,

    /// Frame decimation factor (1 = no decimation, 2 = every 2nd frame, etc.)
    pub decimation: usize,

    /// Buffer size in frames
    pub buffer_size: usize,

    /// Target width (None = original)
    pub target_width: Option<usize>,

    /// Target height (None = original)
    pub target_height: Option<usize>,

    /// Start time in seconds (for seeking)
    pub start_time: Option<f64>,

    /// Maximum frames to read (None = unlimited)
    pub max_frames: Option<u64>,

    /// Camera format (for v4l2/DirectShow/AVFoundation)
    pub camera_format: Option<String>,

    /// Camera FPS (for camera sources)
    pub camera_fps: Option<u32>,

    /// Which decode/capture implementation to use. Defaults to
    /// [`VideoBackend::Auto`]; absent when deserializing an older
    /// [`VideoConfig`] serialization that predates this field.
    #[serde(default)]
    pub backend: VideoBackend,
}

impl VideoConfig {
    /// Create configuration from file path
    pub fn from_file(path: impl Into<PathBuf>) -> Self {
        Self {
            source: VideoSource::File(path.into()),
            pixel_format: PixelFormat::Rgb,
            decimation: 1,
            buffer_size: 30,
            target_width: None,
            target_height: None,
            start_time: None,
            max_frames: None,
            camera_format: None,
            camera_fps: None,
            backend: VideoBackend::Auto,
        }
    }

    /// Create configuration from camera
    pub fn from_camera(device: impl Into<String>) -> Self {
        Self {
            source: VideoSource::Camera(device.into()),
            pixel_format: PixelFormat::Rgb,
            decimation: 1,
            buffer_size: 5,
            target_width: None,
            target_height: None,
            start_time: None,
            max_frames: None,
            // Platform-appropriate default (v4l2 on Linux, dshow on Windows,
            // avfoundation on macOS) rather than hardcoding Linux's.
            camera_format: Some(CameraDevice::default_format().to_string()),
            camera_fps: Some(30),
            backend: VideoBackend::Auto,
        }
    }

    /// Create configuration from network stream
    pub fn from_network(url: impl Into<String>) -> Self {
        Self {
            source: VideoSource::Network(url.into()),
            pixel_format: PixelFormat::Rgb,
            decimation: 1,
            buffer_size: 10,
            target_width: None,
            target_height: None,
            start_time: None,
            max_frames: None,
            camera_format: None,
            camera_fps: None,
            backend: VideoBackend::Auto,
        }
    }

    /// Set pixel format
    pub fn with_pixel_format(mut self, pixel_format: PixelFormat) -> Self {
        self.pixel_format = pixel_format;
        self
    }

    /// Set decimation factor
    pub fn with_decimation(mut self, decimation: usize) -> Self {
        self.decimation = decimation.max(1);
        self
    }

    /// Set buffer size
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }

    /// Set target dimensions
    pub fn with_resize(mut self, width: usize, height: usize) -> Self {
        self.target_width = Some(width);
        self.target_height = Some(height);
        self
    }

    /// Set start time for seeking
    pub fn with_start_time(mut self, start_time: f64) -> Self {
        self.start_time = Some(start_time);
        self
    }

    /// Set maximum frames to read
    pub fn with_max_frames(mut self, max_frames: u64) -> Self {
        self.max_frames = Some(max_frames);
        self
    }

    /// Set camera format (e.g., "video4linux2", "dshow", "avfoundation")
    pub fn with_camera_format(mut self, format: impl Into<String>) -> Self {
        self.camera_format = Some(format.into());
        self
    }

    /// Set camera FPS
    pub fn with_camera_fps(mut self, fps: u32) -> Self {
        self.camera_fps = Some(fps);
        self
    }

    /// Select which decode/capture backend `VideoReader` should use.
    pub fn with_backend(mut self, backend: VideoBackend) -> Self {
        self.backend = backend;
        self
    }
}

/// Camera device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraDevice {
    /// Device path or identifier
    pub path: String,
    /// Device name (if available)
    pub name: Option<String>,
    /// Supported formats
    pub formats: Vec<String>,
}

impl CameraDevice {
    /// List available camera devices, through `oximedia-capture`.
    ///
    /// This is the `video-pure` implementation, and it is used **on every
    /// platform whenever `video-pure` is compiled in** -- including a build
    /// that also has `video`. That is a deliberate preference rather than an
    /// accident of `cfg` ordering: `oximedia-capture` asks the platform
    /// capture API itself (AVFoundation, V4L2, Media Foundation), so it
    /// reports real device names and the modes each one advertises, on
    /// Windows and macOS as well as Linux. The FFmpeg-only implementation it
    /// replaces could do none of that -- it scanned `/dev/video*` on Linux
    /// and returned [`IoError::Unsupported`] everywhere else, because
    /// `ffmpeg-next` does not surface `avdevice_list_input_sources`. A
    /// `video`-only build still gets that older behaviour; see the
    /// `not(feature = "video-pure")` twins below.
    ///
    /// [`CameraDevice::path`] carries the capture backend's own device id --
    /// a `/dev/video*` node on Linux, an `AVCaptureDevice` `uniqueID` on
    /// macOS, a symbolic link on Windows -- which is exactly what
    /// [`VideoConfig::from_camera`] takes. [`CameraDevice::formats`] holds one
    /// label per advertised mode, `"<ENCODING> <W>x<H>@<FPS>"` (for example
    /// `"YUYV 640x480@30"` or `"MJPEG 1280x720@30"`).
    ///
    /// # Errors
    ///
    /// * [`IoError::ConfigError`] with `"No camera devices found"` when the
    ///   backend works but nothing is attached -- the same wording, and the
    ///   same variant, the Linux `/dev` scan has always used for an empty
    ///   list, so existing callers matching on it keep working.
    /// * [`IoError::Connection`] when the platform refuses access (on macOS
    ///   this is the TCC camera permission, and the message says so).
    /// * [`IoError::Unsupported`] on a target `oximedia-capture` has no
    ///   backend for. An empty list is what a *working* backend says when no
    ///   camera is plugged in; conflating that with "not implemented" turns a
    ///   missing feature into a hardware hunt.
    #[cfg(feature = "video-pure")]
    pub fn list_devices() -> IoResult<Vec<CameraDevice>> {
        use tracing::info;

        let listed = oximedia_capture::enumerate().map_err(|error| {
            // The enumeration has no single device to blame, so the mapper is
            // given the selector-shaped label "<any>" rather than a made-up
            // device id.
            super::backend_pure_camera::map_capture_error(error, "<any>")
        })?;

        let devices: Vec<CameraDevice> = listed
            .into_iter()
            .map(|device| CameraDevice {
                path: device.id,
                name: Some(device.name),
                formats: device.formats.iter().map(format_label).collect(),
            })
            .collect();

        if devices.is_empty() {
            Err(IoError::ConfigError("No camera devices found".into()))
        } else {
            info!("Found {} camera device(s)", devices.len());
            Ok(devices)
        }
    }

    /// List available camera devices (FFmpeg-only build, Linux).
    ///
    /// Enumerates `/dev/video*`. See the `video-pure` implementation above
    /// for the richer one that replaces this whenever that feature is
    /// compiled in.
    #[cfg(all(not(feature = "video-pure"), target_os = "linux"))]
    pub fn list_devices() -> IoResult<Vec<CameraDevice>> {
        use std::fs;
        use tracing::info;

        let mut devices: Vec<CameraDevice> = Vec::new();
        if let Ok(entries) = fs::read_dir("/dev") {
            for entry in entries.flatten() {
                if let Ok(file_name) = entry.file_name().into_string() {
                    if file_name.starts_with("video") {
                        let path = format!("/dev/{}", file_name);
                        devices.push(CameraDevice {
                            path,
                            name: Some(file_name),
                            formats: vec!["video4linux2".to_string()],
                        });
                    }
                }
            }
        }

        if devices.is_empty() {
            Err(IoError::ConfigError("No camera devices found".into()))
        } else {
            info!("Found {} camera device(s)", devices.len());
            Ok(devices)
        }
    }

    /// List available camera devices (FFmpeg-only build, non-Linux).
    ///
    /// There is no filesystem enumeration to walk here, and FFmpeg exposes
    /// device listing only through `avdevice_list_input_sources`, which the
    /// safe `ffmpeg-next` API does not surface -- so this returns
    /// [`IoError::Unsupported`]. Previous versions invented a single "Default
    /// Camera" entry at index 0 whether or not any capture device existed; an
    /// honest "cannot enumerate" beats a fabricated device list. Enabling
    /// `video-pure` replaces this with real enumeration.
    #[cfg(all(not(feature = "video-pure"), not(target_os = "linux")))]
    pub fn list_devices() -> IoResult<Vec<CameraDevice>> {
        Err(IoError::Unsupported(format!(
            "Camera enumeration is not implemented on this platform; \
             pass the device identifier directly to \
             VideoConfig::from_camera (input format '{}'), or enable the \
             `video-pure` feature for pure-Rust enumeration",
            Self::default_format()
        )))
    }

    /// Get platform-specific default camera format
    pub fn default_format() -> &'static str {
        #[cfg(target_os = "linux")]
        {
            "video4linux2"
        }
        #[cfg(target_os = "windows")]
        {
            "dshow"
        }
        #[cfg(target_os = "macos")]
        {
            "avfoundation"
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            "auto"
        }
    }
}

/// Human-readable label for one mode a capture device advertises.
///
/// `"<ENCODING> <W>x<H>@<FPS>"` -- `"YUYV 640x480@30"`, `"MJPEG
/// 1280x720@30"`, `"NV12 1920x1080@29.97"`. Deliberately not
/// `CaptureFormat`'s own `Display` (`"640x480@30.000 Yuyv422"`): this is the
/// shape `v4l2-ctl --list-formats-ext` and FFmpeg's device listings print, so
/// a user comparing kizzasi's output against either sees the same strings.
///
/// The frame rate keeps its exact rational form where it has one: a mode
/// advertised as `30/1` prints `30`, and `30000/1001` prints `29.97` rather
/// than a rounded `30`.
#[cfg(feature = "video-pure")]
fn format_label(format: &oximedia_capture::CaptureFormat) -> String {
    let encoding = match format.encoding {
        oximedia_capture::CaptureEncoding::Mjpeg => "MJPEG".to_string(),
        oximedia_capture::CaptureEncoding::Raw(pixel) => pixel_label(pixel),
    };
    // `is_multiple_of(0)` is `self == 0`, which would print "0" for the
    // "unknown / variable rate" case rather than the 0.00 `fps()` reports, so
    // the zero denominator is excluded explicitly.
    let fps = if format.fps_den != 0 && format.fps_num.is_multiple_of(format.fps_den) {
        (format.fps_num / format.fps_den).to_string()
    } else {
        format!("{:.2}", format.fps())
    };
    format!("{encoding} {}x{}@{fps}", format.width, format.height)
}

/// Short uppercase name for a captured pixel layout.
///
/// The well-known FourCC-ish spellings drivers and `v4l2-ctl` use, so the
/// labels line up with what a user sees elsewhere; anything without a
/// customary short name falls back to the enum variant in upper case rather
/// than being dropped.
#[cfg(feature = "video-pure")]
fn pixel_label(pixel: oximedia_core::PixelFormat) -> String {
    use oximedia_core::PixelFormat as Captured;

    match pixel {
        Captured::Yuyv422 => "YUYV".to_string(),
        Captured::Uyvy422 => "UYVY".to_string(),
        Captured::Nv12 => "NV12".to_string(),
        Captured::Nv21 => "NV21".to_string(),
        Captured::Yuv420p => "YU12".to_string(),
        Captured::Yuv422p => "422P".to_string(),
        Captured::Yuv444p => "444P".to_string(),
        Captured::Rgb24 => "RGB3".to_string(),
        Captured::Rgba32 => "RGBA".to_string(),
        Captured::Gray8 => "GREY".to_string(),
        other => format!("{other:?}").to_uppercase(),
    }
}

/// Video metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoMetadata {
    /// Frames per second
    pub fps: f64,

    /// Duration in seconds
    pub duration: f64,

    /// Total frame count
    pub frame_count: u64,

    /// Video width in pixels
    pub width: usize,

    /// Video height in pixels
    pub height: usize,

    /// Pixel format
    pub pixel_format: PixelFormat,
}

/// Video frame buffer for smooth playback
pub struct FrameBuffer {
    frames: Vec<VideoFrame>,
    capacity: usize,
    read_index: usize,
    write_index: usize,
}

impl FrameBuffer {
    /// Create a new frame buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: Vec::with_capacity(capacity),
            capacity,
            read_index: 0,
            write_index: 0,
        }
    }

    /// Push a frame to the buffer
    pub fn push(&mut self, frame: VideoFrame) -> bool {
        if self.frames.len() < self.capacity {
            self.frames.push(frame);
            self.write_index += 1;
            true
        } else {
            // Buffer full
            false
        }
    }

    /// Pop a frame from the buffer
    pub fn pop(&mut self) -> Option<VideoFrame> {
        if self.read_index < self.frames.len() {
            let frame = self.frames[self.read_index].clone();
            self.read_index += 1;

            // Reset if we've consumed all frames
            if self.read_index >= self.frames.len() {
                self.frames.clear();
                self.read_index = 0;
                self.write_index = 0;
            }

            Some(frame)
        } else {
            None
        }
    }

    /// Get number of available frames
    pub fn available(&self) -> usize {
        self.frames.len() - self.read_index
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.frames.len() >= self.capacity
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.read_index >= self.frames.len()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.frames.clear();
        self.read_index = 0;
        self.write_index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::{VideoFilter, VideoProcessor};

    #[test]
    fn test_video_config_file() {
        let config = VideoConfig::from_file("test.mp4")
            .with_decimation(2)
            .with_buffer_size(60)
            .with_resize(640, 480);

        assert_eq!(config.decimation, 2);
        assert_eq!(config.buffer_size, 60);
        assert_eq!(config.target_width, Some(640));
        assert_eq!(config.target_height, Some(480));
    }

    #[test]
    fn test_video_config_camera() {
        let config = VideoConfig::from_camera("/dev/video0").with_pixel_format(PixelFormat::Gray);

        if let VideoSource::Camera(device) = &config.source {
            assert_eq!(device, "/dev/video0");
        } else {
            panic!("Expected Camera source");
        }

        assert_eq!(config.pixel_format, PixelFormat::Gray);
    }

    #[test]
    fn test_frame_buffer() {
        let mut buffer = FrameBuffer::new(3);

        assert!(buffer.is_empty());
        assert!(!buffer.is_full());

        // Add frames
        for i in 0..3 {
            let frame = VideoFrame {
                index: i,
                timestamp: i as f64 / 30.0,
                width: 640,
                height: 480,
                channels: 3,
                data: vec![0; 640 * 480 * 3],
            };
            assert!(buffer.push(frame));
        }

        assert!(buffer.is_full());
        assert_eq!(buffer.available(), 3);

        // Pop frames
        for i in 0..3 {
            let frame = buffer.pop().unwrap();
            assert_eq!(frame.index, i);
        }

        assert!(buffer.is_empty());
    }

    #[test]
    fn test_frame_to_grayscale() {
        let frame = VideoFrame {
            index: 0,
            timestamp: 0.0,
            width: 2,
            height: 2,
            channels: 3,
            data: vec![
                255, 0, 0, // Red
                0, 255, 0, // Green
                0, 0, 255, // Blue
                255, 255, 255, // White
            ],
        };

        let gray = frame.to_grayscale().expect("valid RGB frame");
        assert_eq!(gray.channels, 1);
        assert_eq!(gray.data.len(), 4);

        // Check gray values (approximate)
        assert!(gray.data[0] > 50 && gray.data[0] < 100); // Red
        assert!(gray.data[1] > 140 && gray.data[1] < 160); // Green
        assert!(gray.data[2] > 20 && gray.data[2] < 40); // Blue
        assert_eq!(gray.data[3], 255); // White
    }

    #[test]
    fn test_frame_normalized() {
        let frame = VideoFrame {
            index: 0,
            timestamp: 0.0,
            width: 2,
            height: 1,
            channels: 1,
            data: vec![0, 128, 255],
        };

        let normalized = frame.to_normalized_f32();
        assert_eq!(normalized.len(), 3);
        assert!((normalized[0] - 0.0).abs() < 0.01);
        assert!((normalized[1] - 0.502).abs() < 0.01);
        assert!((normalized[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pixel_format_channels() {
        assert_eq!(PixelFormat::Gray.channels(), 1);
        assert_eq!(PixelFormat::Rgb.channels(), 3);
        assert_eq!(PixelFormat::Rgba.channels(), 4);
    }

    // ================================================================
    // Robustness: user-controlled frame data must never panic (id=47)
    // ================================================================

    #[test]
    fn test_frame_with_mismatched_data_length_errors_instead_of_panicking() {
        let frame = VideoFrame {
            index: 0,
            timestamp: 0.0,
            width: 4,
            height: 4,
            channels: 3,
            data: vec![0; 7], // != 4 * 4 * 3
        };

        assert!(frame.validate().is_err());
        assert!(frame.to_array().is_err());
        assert!(frame.as_array().is_err());
        assert!(frame.to_grayscale().is_err());
        assert!(VideoProcessor::apply_filter(&frame, VideoFilter::SobelEdge).is_err());
    }

    #[test]
    fn test_two_channel_frame_is_rejected_not_read_out_of_bounds() {
        let frame = VideoFrame {
            index: 0,
            timestamp: 0.0,
            width: 2,
            height: 1,
            channels: 2,
            data: vec![10, 20, 30, 40],
        };
        // Previously this indexed data[idx + 2] unconditionally.
        assert!(matches!(frame.to_grayscale(), Err(IoError::Unsupported(_))));
    }

    #[test]
    fn test_zero_channel_frame_is_rejected() {
        let frame = VideoFrame {
            index: 0,
            timestamp: 0.0,
            width: 2,
            height: 2,
            channels: 0,
            data: Vec::new(),
        };
        assert!(frame.validate().is_err());
    }

    #[test]
    fn test_video_config_camera_default_format_matches_platform_default() {
        // Regression test: `from_camera` used to hardcode "video4linux2" as
        // the default camera_format on every OS, even though
        // `CameraDevice::default_format()` is platform-gated (dshow on
        // Windows, avfoundation on macOS). The two must agree.
        let config = VideoConfig::from_camera("some-device");
        assert_eq!(
            config.camera_format.as_deref(),
            Some(CameraDevice::default_format())
        );
    }
}
