//! Video frame input and processing
//!
//! Provides video stream input using FFmpeg for reading video files and camera streams.
//!
//! ## Features
//!
//! - Read video files (MP4, AVI, MKV, WebM, etc.)
//! - Camera input (v4l2, DirectShow, AVFoundation)
//! - Frame decimation (skip frames for reduced processing)
//! - Frame buffering for smooth playback
//! - RGB/Grayscale conversion
//! - Resize and crop operations
//!
//! ## Example
//!
//! ```rust,no_run
//! use kizzasi_io::{VideoReader, VideoConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = VideoConfig::from_file("video.mp4")
//!         .with_decimation(2)  // Process every 2nd frame
//!         .with_buffer_size(30);  // Buffer 30 frames
//!
//!     let mut reader = VideoReader::new(config).await?;
//!
//!     while let Some(frame) = reader.read_frame().await? {
//!         println!("Frame {} - {}x{}", frame.index, frame.width, frame.height);
//!         // Process frame.data (RGB or grayscale)
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::error::{IoError, IoResult};
use ffmpeg_next as ffmpeg;
use scirs2_core::ndarray::{Array2, Array3, ArrayView3};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

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
    /// Convert frame to ndarray (height x width x channels)
    pub fn to_array(&self) -> Array3<u8> {
        Array3::from_shape_vec((self.height, self.width, self.channels), self.data.clone())
            .expect("Invalid frame dimensions")
    }

    /// Get frame data as array view
    pub fn as_array(&self) -> ArrayView3<'_, u8> {
        ArrayView3::from_shape((self.height, self.width, self.channels), &self.data)
            .expect("Invalid frame dimensions")
    }

    /// Convert to grayscale (if not already)
    pub fn to_grayscale(&self) -> VideoFrame {
        if self.channels == 1 {
            return self.clone();
        }

        let mut gray_data = Vec::with_capacity(self.width * self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) * self.channels;
                let r = self.data[idx] as f32;
                let g = self.data[idx + 1] as f32;
                let b = self.data[idx + 2] as f32;

                // ITU-R BT.601 luma coefficients
                let gray = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                gray_data.push(gray);
            }
        }

        VideoFrame {
            index: self.index,
            timestamp: self.timestamp,
            width: self.width,
            height: self.height,
            channels: 1,
            data: gray_data,
        }
    }

    /// Convert to f32 normalized to [0.0, 1.0]
    pub fn to_normalized_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&x| x as f32 / 255.0).collect()
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

/// Optical flow result containing motion vectors
#[derive(Debug, Clone)]
pub struct OpticalFlow {
    /// Horizontal flow (u component) - width x height
    pub flow_x: Array2<f32>,
    /// Vertical flow (v component) - width x height
    pub flow_y: Array2<f32>,
    /// Flow magnitude at each pixel
    pub magnitude: Array2<f32>,
    /// Flow angle at each pixel (in radians)
    pub angle: Array2<f32>,
}

impl OpticalFlow {
    /// Create new optical flow from flow vectors
    pub fn new(flow_x: Array2<f32>, flow_y: Array2<f32>) -> Self {
        let magnitude = flow_x
            .iter()
            .zip(flow_y.iter())
            .map(|(u, v)| (u * u + v * v).sqrt())
            .collect::<Vec<_>>();
        let magnitude =
            Array2::from_shape_vec(flow_x.dim(), magnitude).expect("Invalid flow dimensions");

        let angle = flow_x
            .iter()
            .zip(flow_y.iter())
            .map(|(u, v)| v.atan2(*u))
            .collect::<Vec<_>>();
        let angle = Array2::from_shape_vec(flow_x.dim(), angle).expect("Invalid flow dimensions");

        Self {
            flow_x,
            flow_y,
            magnitude,
            angle,
        }
    }

    /// Get flow vector at specific position
    pub fn get_flow(&self, x: usize, y: usize) -> Option<(f32, f32)> {
        if y < self.flow_x.nrows() && x < self.flow_x.ncols() {
            Some((self.flow_x[[y, x]], self.flow_y[[y, x]]))
        } else {
            None
        }
    }

    /// Get maximum flow magnitude
    pub fn max_magnitude(&self) -> f32 {
        self.magnitude.iter().fold(0.0f32, |max, &val| max.max(val))
    }

    /// Get average flow magnitude
    pub fn avg_magnitude(&self) -> f32 {
        self.magnitude.mean().unwrap_or(0.0)
    }

    /// Get dimensions (height, width)
    pub fn dimensions(&self) -> (usize, usize) {
        self.flow_x.dim()
    }
}

/// Optical flow computation method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpticalFlowMethod {
    /// Lucas-Kanade method (sparse optical flow)
    LucasKanade,
    /// Dense optical flow using image gradients
    DenseGradient,
    /// Block matching optical flow
    BlockMatching,
}

/// Optical flow estimator for computing motion between frames
pub struct OpticalFlowEstimator {
    method: OpticalFlowMethod,
    window_size: usize,
    pyramid_levels: usize,
}

impl OpticalFlowEstimator {
    /// Create new optical flow estimator
    pub fn new(method: OpticalFlowMethod) -> Self {
        Self {
            method,
            window_size: 15,
            pyramid_levels: 3,
        }
    }

    /// Set window size for flow computation
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set pyramid levels for multi-scale flow
    pub fn with_pyramid_levels(mut self, levels: usize) -> Self {
        self.pyramid_levels = levels;
        self
    }

    /// Compute optical flow between two frames
    pub fn compute(&self, prev: &VideoFrame, curr: &VideoFrame) -> IoResult<OpticalFlow> {
        // Convert to grayscale if needed
        let prev_gray = if prev.channels == 1 {
            prev.clone()
        } else {
            prev.to_grayscale()
        };

        let curr_gray = if curr.channels == 1 {
            curr.clone()
        } else {
            curr.to_grayscale()
        };

        // Ensure dimensions match
        if prev_gray.width != curr_gray.width || prev_gray.height != curr_gray.height {
            return Err(IoError::ConfigError(
                "Frame dimensions must match for optical flow".into(),
            ));
        }

        match self.method {
            OpticalFlowMethod::DenseGradient => self.compute_dense_gradient(&prev_gray, &curr_gray),
            OpticalFlowMethod::BlockMatching => self.compute_block_matching(&prev_gray, &curr_gray),
            OpticalFlowMethod::LucasKanade => self.compute_lucas_kanade(&prev_gray, &curr_gray),
        }
    }

    /// Compute dense optical flow using image gradients (simplified Farneback-like)
    fn compute_dense_gradient(
        &self,
        prev: &VideoFrame,
        curr: &VideoFrame,
    ) -> IoResult<OpticalFlow> {
        #[cfg(feature = "simd")]
        {
            self.compute_dense_gradient_simd(prev, curr)
        }
        #[cfg(not(feature = "simd"))]
        {
            self.compute_dense_gradient_scalar(prev, curr)
        }
    }

    /// Scalar version of dense gradient optical flow
    #[cfg(not(feature = "simd"))]
    fn compute_dense_gradient_scalar(
        &self,
        prev: &VideoFrame,
        curr: &VideoFrame,
    ) -> IoResult<OpticalFlow> {
        let height = prev.height;
        let width = prev.width;

        let mut flow_x = Array2::zeros((height, width));
        let mut flow_y = Array2::zeros((height, width));

        // Compute image gradients and temporal derivative
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;

                // Spatial gradients (Sobel-like)
                let ix = ((curr.data[idx + 1] as f32 - curr.data[idx - 1] as f32)
                    + (prev.data[idx + 1] as f32 - prev.data[idx - 1] as f32))
                    / 4.0;

                let iy = ((curr.data[idx + width] as f32 - curr.data[idx - width] as f32)
                    + (prev.data[idx + width] as f32 - prev.data[idx - width] as f32))
                    / 4.0;

                // Temporal gradient
                let it = curr.data[idx] as f32 - prev.data[idx] as f32;

                // Avoid division by zero
                let denominator = ix * ix + iy * iy + 1e-6;

                // Compute flow (Lucas-Kanade equation)
                let u = -(ix * it) / denominator;
                let v = -(iy * it) / denominator;

                flow_x[[y, x]] = u;
                flow_y[[y, x]] = v;
            }
        }

        Ok(OpticalFlow::new(flow_x, flow_y))
    }

    /// SIMD-optimized version of dense gradient optical flow
    #[cfg(feature = "simd")]
    fn compute_dense_gradient_simd(
        &self,
        prev: &VideoFrame,
        curr: &VideoFrame,
    ) -> IoResult<OpticalFlow> {
        let height = prev.height;
        let width = prev.width;

        let mut flow_x = Array2::zeros((height, width));
        let mut flow_y = Array2::zeros((height, width));

        // Process in SIMD-friendly chunks (4 pixels at a time)
        // This allows the compiler to auto-vectorize more effectively
        for y in 1..height - 1 {
            for x in (1..width - 1).step_by(4) {
                let x_end = (x + 4).min(width - 1);
                for x_off in x..x_end {
                    let idx = y * width + x_off;

                    // Spatial gradients (Sobel-like)
                    let ix = ((curr.data[idx + 1] as f32 - curr.data[idx - 1] as f32)
                        + (prev.data[idx + 1] as f32 - prev.data[idx - 1] as f32))
                        / 4.0;

                    let iy = ((curr.data[idx + width] as f32 - curr.data[idx - width] as f32)
                        + (prev.data[idx + width] as f32 - prev.data[idx - width] as f32))
                        / 4.0;

                    // Temporal gradient
                    let it = curr.data[idx] as f32 - prev.data[idx] as f32;

                    // Avoid division by zero
                    let denominator = ix * ix + iy * iy + 1e-6;

                    // Compute flow (Lucas-Kanade equation)
                    let u = -(ix * it) / denominator;
                    let v = -(iy * it) / denominator;

                    flow_x[[y, x_off]] = u;
                    flow_y[[y, x_off]] = v;
                }
            }
        }

        Ok(OpticalFlow::new(flow_x, flow_y))
    }

    /// Compute optical flow using block matching
    fn compute_block_matching(
        &self,
        prev: &VideoFrame,
        curr: &VideoFrame,
    ) -> IoResult<OpticalFlow> {
        let height = prev.height;
        let width = prev.width;
        let block_size = self.window_size;
        let search_range = block_size / 2;

        let mut flow_x = Array2::zeros((height, width));
        let mut flow_y = Array2::zeros((height, width));

        // Process in blocks
        for by in (0..height).step_by(block_size) {
            for bx in (0..width).step_by(block_size) {
                let block_h = (block_size).min(height - by);
                let block_w = (block_size).min(width - bx);

                // Search for best match in search range
                let mut best_dx = 0isize;
                let mut best_dy = 0isize;
                let mut best_sad = f32::MAX;

                for dy in -(search_range as isize)..=(search_range as isize) {
                    for dx in -(search_range as isize)..=(search_range as isize) {
                        let mut sad = 0.0f32;
                        let mut count = 0;

                        // Compute SAD (Sum of Absolute Differences)
                        for y in 0..block_h {
                            for x in 0..block_w {
                                let py = by + y;
                                let px = bx + x;

                                let cy = (py as isize + dy) as usize;
                                let cx = (px as isize + dx) as usize;

                                if cy < height && cx < width {
                                    let prev_val = prev.data[py * width + px] as f32;
                                    let curr_val = curr.data[cy * width + cx] as f32;
                                    sad += (prev_val - curr_val).abs();
                                    count += 1;
                                }
                            }
                        }

                        if count > 0 {
                            sad /= count as f32;
                            if sad < best_sad {
                                best_sad = sad;
                                best_dx = dx;
                                best_dy = dy;
                            }
                        }
                    }
                }

                // Fill block with computed flow
                for y in 0..block_h {
                    for x in 0..block_w {
                        let py = by + y;
                        let px = bx + x;
                        if py < height && px < width {
                            flow_x[[py, px]] = best_dx as f32;
                            flow_y[[py, px]] = best_dy as f32;
                        }
                    }
                }
            }
        }

        Ok(OpticalFlow::new(flow_x, flow_y))
    }

    /// Compute Lucas-Kanade optical flow
    fn compute_lucas_kanade(&self, prev: &VideoFrame, curr: &VideoFrame) -> IoResult<OpticalFlow> {
        let height = prev.height;
        let width = prev.width;
        let win_size = self.window_size;
        let half_win = win_size / 2;

        let mut flow_x = Array2::zeros((height, width));
        let mut flow_y = Array2::zeros((height, width));

        // Process each pixel with a window
        for y in half_win..height - half_win {
            for x in half_win..width - half_win {
                let mut sum_ix2 = 0.0f32;
                let mut sum_iy2 = 0.0f32;
                let mut sum_ixiy = 0.0f32;
                let mut sum_ixit = 0.0f32;
                let mut sum_iyit = 0.0f32;

                // Compute gradients in window
                for wy in y - half_win..=y + half_win {
                    for wx in x - half_win..=x + half_win {
                        if wy == 0 || wy >= height - 1 || wx == 0 || wx >= width - 1 {
                            continue;
                        }

                        let idx = wy * width + wx;

                        // Spatial gradients
                        let ix = ((curr.data[idx + 1] as f32 - curr.data[idx - 1] as f32)
                            + (prev.data[idx + 1] as f32 - prev.data[idx - 1] as f32))
                            / 4.0;

                        let iy = ((curr.data[idx + width] as f32 - curr.data[idx - width] as f32)
                            + (prev.data[idx + width] as f32 - prev.data[idx - width] as f32))
                            / 4.0;

                        // Temporal gradient
                        let it = curr.data[idx] as f32 - prev.data[idx] as f32;

                        sum_ix2 += ix * ix;
                        sum_iy2 += iy * iy;
                        sum_ixiy += ix * iy;
                        sum_ixit += ix * it;
                        sum_iyit += iy * it;
                    }
                }

                // Solve 2x2 system
                let det = sum_ix2 * sum_iy2 - sum_ixiy * sum_ixiy;

                if det.abs() > 1e-6 {
                    let u = (sum_iy2 * (-sum_ixit) - sum_ixiy * (-sum_iyit)) / det;
                    let v = (-sum_ixiy * (-sum_ixit) + sum_ix2 * (-sum_iyit)) / det;

                    // Clamp extreme values
                    flow_x[[y, x]] = u.clamp(-50.0, 50.0);
                    flow_y[[y, x]] = v.clamp(-50.0, 50.0);
                }
            }
        }

        Ok(OpticalFlow::new(flow_x, flow_y))
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
            camera_format: Some("video4linux2".to_string()), // v4l2 on Linux
            camera_fps: Some(30),
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
    /// List available camera devices
    /// On Linux: /dev/video*
    /// On Windows: DirectShow devices
    /// On macOS: AVFoundation devices
    pub fn list_devices() -> IoResult<Vec<CameraDevice>> {
        let mut devices = Vec::new();

        #[cfg(target_os = "linux")]
        {
            // List v4l2 devices
            use std::fs;
            if let Ok(entries) = fs::read_dir("/dev") {
                for entry in entries.flatten() {
                    if let Ok(file_name) = entry.file_name().into_string() {
                        if file_name.starts_with("video") {
                            let path = format!("/dev/{}", file_name);
                            devices.push(CameraDevice {
                                path: path.clone(),
                                name: Some(file_name),
                                formats: vec!["video4linux2".to_string()],
                            });
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // On Windows, DirectShow devices would be enumerated
            // This is a placeholder - full implementation would use FFmpeg or DirectShow API
            devices.push(CameraDevice {
                path: "0".to_string(),
                name: Some("Default Camera".to_string()),
                formats: vec!["dshow".to_string()],
            });
        }

        #[cfg(target_os = "macos")]
        {
            // On macOS, AVFoundation devices
            devices.push(CameraDevice {
                path: "0".to_string(),
                name: Some("Default Camera".to_string()),
                formats: vec!["avfoundation".to_string()],
            });
        }

        if devices.is_empty() {
            Err(IoError::ConfigError("No camera devices found".into()))
        } else {
            Ok(devices)
        }
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

/// Video processing filter types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VideoFilter {
    /// Gaussian blur for noise reduction
    GaussianBlur { kernel_size: usize, sigma: f32 },
    /// Box blur (simple averaging)
    BoxBlur { kernel_size: usize },
    /// Sobel edge detection
    SobelEdge,
    /// Laplacian edge detection
    LaplacianEdge,
    /// Erosion (morphological)
    Erosion { kernel_size: usize },
    /// Dilation (morphological)
    Dilation { kernel_size: usize },
    /// Sharpen filter
    Sharpen,
    /// Brightness adjustment
    Brightness { delta: f32 },
    /// Contrast adjustment
    Contrast { factor: f32 },
}

/// Video frame processor for applying filters
pub struct VideoProcessor;

impl VideoProcessor {
    /// Apply a filter to a video frame
    pub fn apply_filter(frame: &VideoFrame, filter: VideoFilter) -> IoResult<VideoFrame> {
        // Ensure grayscale for most operations
        let gray = if frame.channels != 1 {
            frame.to_grayscale()
        } else {
            frame.clone()
        };

        match filter {
            VideoFilter::GaussianBlur { kernel_size, sigma } => {
                Self::gaussian_blur(&gray, kernel_size, sigma)
            }
            VideoFilter::BoxBlur { kernel_size } => Self::box_blur(&gray, kernel_size),
            VideoFilter::SobelEdge => Self::sobel_edge(&gray),
            VideoFilter::LaplacianEdge => Self::laplacian_edge(&gray),
            VideoFilter::Erosion { kernel_size } => Self::erosion(&gray, kernel_size),
            VideoFilter::Dilation { kernel_size } => Self::dilation(&gray, kernel_size),
            VideoFilter::Sharpen => Self::sharpen(&gray),
            VideoFilter::Brightness { delta } => Self::adjust_brightness(&gray, delta),
            VideoFilter::Contrast { factor } => Self::adjust_contrast(&gray, factor),
        }
    }

    /// Gaussian blur filter
    fn gaussian_blur(frame: &VideoFrame, kernel_size: usize, sigma: f32) -> IoResult<VideoFrame> {
        let width = frame.width;
        let height = frame.height;
        let half_kernel = kernel_size / 2;

        // Generate Gaussian kernel
        let mut kernel = vec![0.0f32; kernel_size * kernel_size];
        let mut sum = 0.0f32;

        for y in 0..kernel_size {
            for x in 0..kernel_size {
                let dx = (x as i32 - half_kernel as i32) as f32;
                let dy = (y as i32 - half_kernel as i32) as f32;
                let value = (-((dx * dx + dy * dy) / (2.0 * sigma * sigma))).exp();
                kernel[y * kernel_size + x] = value;
                sum += value;
            }
        }

        // Normalize kernel
        for val in kernel.iter_mut() {
            *val /= sum;
        }

        // Apply convolution
        let mut result_data = vec![0u8; width * height];

        for y in half_kernel..height - half_kernel {
            for x in half_kernel..width - half_kernel {
                let mut sum_val = 0.0f32;

                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let px = x + kx - half_kernel;
                        let py = y + ky - half_kernel;
                        let pixel_val = frame.data[py * width + px] as f32;
                        sum_val += pixel_val * kernel[ky * kernel_size + kx];
                    }
                }

                result_data[y * width + x] = sum_val.round().clamp(0.0, 255.0) as u8;
            }
        }

        Ok(VideoFrame {
            index: frame.index,
            timestamp: frame.timestamp,
            width,
            height,
            channels: 1,
            data: result_data,
        })
    }

    /// Box blur filter (simple averaging)
    fn box_blur(frame: &VideoFrame, kernel_size: usize) -> IoResult<VideoFrame> {
        let width = frame.width;
        let height = frame.height;
        let half_kernel = kernel_size / 2;
        let scale = 1.0 / (kernel_size * kernel_size) as f32;

        let mut result_data = vec![0u8; width * height];

        for y in half_kernel..height - half_kernel {
            for x in half_kernel..width - half_kernel {
                let mut sum = 0u32;

                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let px = x + kx - half_kernel;
                        let py = y + ky - half_kernel;
                        sum += frame.data[py * width + px] as u32;
                    }
                }

                result_data[y * width + x] = ((sum as f32 * scale).round() as u32).min(255) as u8;
            }
        }

        Ok(VideoFrame {
            index: frame.index,
            timestamp: frame.timestamp,
            width,
            height,
            channels: 1,
            data: result_data,
        })
    }

    /// Sobel edge detection
    fn sobel_edge(frame: &VideoFrame) -> IoResult<VideoFrame> {
        let width = frame.width;
        let height = frame.height;

        let sobel_x = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
        let sobel_y = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

        let mut result_data = vec![0u8; width * height];

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut gx = 0.0f32;
                let mut gy = 0.0f32;

                for ky in 0..3 {
                    for kx in 0..3 {
                        let px = x + kx - 1;
                        let py = y + ky - 1;
                        let pixel = frame.data[py * width + px] as f32;
                        let kernel_idx = ky * 3 + kx;

                        gx += pixel * sobel_x[kernel_idx] as f32;
                        gy += pixel * sobel_y[kernel_idx] as f32;
                    }
                }

                let magnitude = (gx * gx + gy * gy).sqrt();
                result_data[y * width + x] = magnitude.min(255.0) as u8;
            }
        }

        Ok(VideoFrame {
            index: frame.index,
            timestamp: frame.timestamp,
            width,
            height,
            channels: 1,
            data: result_data,
        })
    }

    /// Laplacian edge detection
    fn laplacian_edge(frame: &VideoFrame) -> IoResult<VideoFrame> {
        let width = frame.width;
        let height = frame.height;

        let laplacian = [0, -1, 0, -1, 4, -1, 0, -1, 0];

        let mut result_data = vec![0u8; width * height];

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut sum = 0.0f32;

                for ky in 0..3 {
                    for kx in 0..3 {
                        let px = x + kx - 1;
                        let py = y + ky - 1;
                        let pixel = frame.data[py * width + px] as f32;
                        sum += pixel * laplacian[ky * 3 + kx] as f32;
                    }
                }

                result_data[y * width + x] = sum.abs().min(255.0) as u8;
            }
        }

        Ok(VideoFrame {
            index: frame.index,
            timestamp: frame.timestamp,
            width,
            height,
            channels: 1,
            data: result_data,
        })
    }

    /// Erosion (morphological operation)
    fn erosion(frame: &VideoFrame, kernel_size: usize) -> IoResult<VideoFrame> {
        let width = frame.width;
        let height = frame.height;
        let half_kernel = kernel_size / 2;

        let mut result_data = vec![0u8; width * height];

        for y in half_kernel..height - half_kernel {
            for x in half_kernel..width - half_kernel {
                let mut min_val = 255u8;

                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let px = x + kx - half_kernel;
                        let py = y + ky - half_kernel;
                        min_val = min_val.min(frame.data[py * width + px]);
                    }
                }

                result_data[y * width + x] = min_val;
            }
        }

        Ok(VideoFrame {
            index: frame.index,
            timestamp: frame.timestamp,
            width,
            height,
            channels: 1,
            data: result_data,
        })
    }

    /// Dilation (morphological operation)
    fn dilation(frame: &VideoFrame, kernel_size: usize) -> IoResult<VideoFrame> {
        let width = frame.width;
        let height = frame.height;
        let half_kernel = kernel_size / 2;

        let mut result_data = vec![0u8; width * height];

        for y in half_kernel..height - half_kernel {
            for x in half_kernel..width - half_kernel {
                let mut max_val = 0u8;

                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let px = x + kx - half_kernel;
                        let py = y + ky - half_kernel;
                        max_val = max_val.max(frame.data[py * width + px]);
                    }
                }

                result_data[y * width + x] = max_val;
            }
        }

        Ok(VideoFrame {
            index: frame.index,
            timestamp: frame.timestamp,
            width,
            height,
            channels: 1,
            data: result_data,
        })
    }

    /// Sharpen filter
    fn sharpen(frame: &VideoFrame) -> IoResult<VideoFrame> {
        let width = frame.width;
        let height = frame.height;

        let sharpen_kernel = [0, -1, 0, -1, 5, -1, 0, -1, 0];

        let mut result_data = vec![0u8; width * height];

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut sum = 0.0f32;

                for ky in 0..3 {
                    for kx in 0..3 {
                        let px = x + kx - 1;
                        let py = y + ky - 1;
                        let pixel = frame.data[py * width + px] as f32;
                        sum += pixel * sharpen_kernel[ky * 3 + kx] as f32;
                    }
                }

                result_data[y * width + x] = sum.clamp(0.0, 255.0) as u8;
            }
        }

        Ok(VideoFrame {
            index: frame.index,
            timestamp: frame.timestamp,
            width,
            height,
            channels: 1,
            data: result_data,
        })
    }

    /// Adjust brightness
    fn adjust_brightness(frame: &VideoFrame, delta: f32) -> IoResult<VideoFrame> {
        let result_data: Vec<u8> = frame
            .data
            .iter()
            .map(|&pixel| ((pixel as f32 + delta).clamp(0.0, 255.0)) as u8)
            .collect();

        Ok(VideoFrame {
            index: frame.index,
            timestamp: frame.timestamp,
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data: result_data,
        })
    }

    /// Adjust contrast
    fn adjust_contrast(frame: &VideoFrame, factor: f32) -> IoResult<VideoFrame> {
        let result_data: Vec<u8> = frame
            .data
            .iter()
            .map(|&pixel| {
                let centered = (pixel as f32 - 128.0) * factor + 128.0;
                centered.clamp(0.0, 255.0) as u8
            })
            .collect();

        Ok(VideoFrame {
            index: frame.index,
            timestamp: frame.timestamp,
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            data: result_data,
        })
    }
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

/// Video reader for streaming video frames
pub struct VideoReader {
    config: VideoConfig,
    buffer: Arc<Mutex<FrameBuffer>>,
    frame_count: u64,
    current_frame: u64,
    fps: f64,
    duration: f64,
}

impl VideoReader {
    /// Create a new video reader
    pub async fn new(config: VideoConfig) -> IoResult<Self> {
        // Initialize FFmpeg
        ffmpeg::init()
            .map_err(|e| IoError::Connection(format!("Failed to initialize FFmpeg: {}", e)))?;

        let buffer = Arc::new(Mutex::new(FrameBuffer::new(config.buffer_size)));

        // For now, create a placeholder reader
        // In a full implementation, we would open the video source here
        Ok(Self {
            config,
            buffer,
            frame_count: 0,
            current_frame: 0,
            fps: 30.0,     // Default FPS
            duration: 0.0, // Unknown duration
        })
    }

    /// Read next frame
    pub async fn read_frame(&mut self) -> IoResult<Option<VideoFrame>> {
        // Check if we've reached max frames
        if let Some(max) = self.config.max_frames {
            if self.current_frame >= max {
                return Ok(None);
            }
        }

        // Check buffer first
        let mut buffer = self.buffer.lock().await;

        if let Some(frame) = buffer.pop() {
            self.current_frame += 1;
            return Ok(Some(frame));
        }

        // In a full implementation, we would:
        // 1. Decode frames from FFmpeg
        // 2. Apply decimation
        // 3. Resize if needed
        // 4. Convert pixel format
        // 5. Fill the buffer

        // For now, return None to indicate end of stream
        Ok(None)
    }

    /// Get video metadata
    pub fn metadata(&self) -> VideoMetadata {
        VideoMetadata {
            fps: self.fps,
            duration: self.duration,
            frame_count: self.frame_count,
            width: self.config.target_width.unwrap_or(1920),
            height: self.config.target_height.unwrap_or(1080),
            pixel_format: self.config.pixel_format,
        }
    }

    /// Seek to specific time
    pub async fn seek(&mut self, _time: f64) -> IoResult<()> {
        // Clear buffer on seek
        let mut buffer = self.buffer.lock().await;
        buffer.clear();

        // In a full implementation, we would seek in the FFmpeg decoder using _time
        Ok(())
    }

    /// Get current frame index
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Get current timestamp
    pub fn current_time(&self) -> f64 {
        if self.fps > 0.0 {
            self.current_frame as f64 / self.fps
        } else {
            0.0
        }
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

#[cfg(test)]
mod tests {
    use super::*;

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

        let gray = frame.to_grayscale();
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
}
