//! Video frame processing: optical flow estimation and spatial filters.
//!
//! This is a split of the former `video.rs`; see the parent `video` module
//! for the module-level documentation.

use super::types::VideoFrame;
use crate::error::{IoError, IoResult};
use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};

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
    /// Create new optical flow from flow vectors.
    ///
    /// Returns [`IoError::ConfigError`] when the two components have
    /// different shapes; the previous version zipped the two iterators (which
    /// silently truncates to the shorter one) and then `.expect()`ed the
    /// reshape, aborting the process on mismatched input.
    pub fn new(flow_x: Array2<f32>, flow_y: Array2<f32>) -> IoResult<Self> {
        if flow_x.dim() != flow_y.dim() {
            return Err(IoError::ConfigError(format!(
                "Optical flow components must have the same shape: {:?} vs {:?}",
                flow_x.dim(),
                flow_y.dim()
            )));
        }

        let magnitude = flow_x
            .iter()
            .zip(flow_y.iter())
            .map(|(u, v)| (u * u + v * v).sqrt())
            .collect::<Vec<_>>();
        let magnitude = Array2::from_shape_vec(flow_x.dim(), magnitude)
            .map_err(|e| IoError::ConfigError(format!("Invalid flow dimensions: {e}")))?;

        let angle = flow_x
            .iter()
            .zip(flow_y.iter())
            .map(|(u, v)| v.atan2(*u))
            .collect::<Vec<_>>();
        let angle = Array2::from_shape_vec(flow_x.dim(), angle)
            .map_err(|e| IoError::ConfigError(format!("Invalid flow dimensions: {e}")))?;

        Ok(Self {
            flow_x,
            flow_y,
            magnitude,
            angle,
        })
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
        // Convert to grayscale if needed (validates both frames)
        let prev_gray = prev.to_grayscale()?;
        let curr_gray = curr.to_grayscale()?;

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

    /// Compute dense optical flow using image gradients (simplified Farneback-like).
    ///
    /// There used to be a second, `#[cfg(feature = "simd")]` copy of this
    /// function whose only difference was an outer `step_by(4)` loop that
    /// still walked the same four pixels one at a time -- byte-for-byte the
    /// same arithmetic under a name promising vectorisation. It has been
    /// removed; this single implementation is always compiled.
    ///
    /// Frames narrower or shorter than 3 pixels have no interior, so the flow
    /// field is all zeros (the same value the border pixels of any frame get).
    fn compute_dense_gradient(
        &self,
        prev: &VideoFrame,
        curr: &VideoFrame,
    ) -> IoResult<OpticalFlow> {
        let height = prev.height;
        let width = prev.width;

        let mut flow_x = Array2::zeros((height, width));
        let mut flow_y = Array2::zeros((height, width));

        // `saturating_sub` keeps the range empty (rather than wrapping to
        // usize::MAX and indexing out of bounds) for 0/1/2-pixel frames.
        for y in 1..height.saturating_sub(1) {
            for x in 1..width.saturating_sub(1) {
                let idx = y * width + x;

                // Spatial gradients (Sobel-like)
                let ix = ((f32::from(curr.data[idx + 1]) - f32::from(curr.data[idx - 1]))
                    + (f32::from(prev.data[idx + 1]) - f32::from(prev.data[idx - 1])))
                    / 4.0;

                let iy = ((f32::from(curr.data[idx + width]) - f32::from(curr.data[idx - width]))
                    + (f32::from(prev.data[idx + width]) - f32::from(prev.data[idx - width])))
                    / 4.0;

                // Temporal gradient
                let it = f32::from(curr.data[idx]) - f32::from(prev.data[idx]);

                // Avoid division by zero
                let denominator = ix * ix + iy * iy + 1e-6;

                // Compute flow (Lucas-Kanade equation)
                let u = -(ix * it) / denominator;
                let v = -(iy * it) / denominator;

                flow_x[[y, x]] = u;
                flow_y[[y, x]] = v;
            }
        }

        OpticalFlow::new(flow_x, flow_y)
    }

    /// Compute optical flow using block matching
    fn compute_block_matching(
        &self,
        prev: &VideoFrame,
        curr: &VideoFrame,
    ) -> IoResult<OpticalFlow> {
        let height = prev.height;
        let width = prev.width;
        // `step_by(0)` panics, so a zero window size degenerates to 1x1 blocks.
        let block_size = self.window_size.max(1);
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

        OpticalFlow::new(flow_x, flow_y)
    }

    /// Compute Lucas-Kanade optical flow
    fn compute_lucas_kanade(&self, prev: &VideoFrame, curr: &VideoFrame) -> IoResult<OpticalFlow> {
        let height = prev.height;
        let width = prev.width;
        let win_size = self.window_size;
        let half_win = win_size / 2;

        let mut flow_x = Array2::zeros((height, width));
        let mut flow_y = Array2::zeros((height, width));

        // Process each pixel with a window. `saturating_sub` keeps the range
        // empty for frames smaller than the window instead of wrapping to
        // usize::MAX and indexing out of bounds.
        for y in half_win..height.saturating_sub(half_win) {
            for x in half_win..width.saturating_sub(half_win) {
                let mut sum_ix2 = 0.0f32;
                let mut sum_iy2 = 0.0f32;
                let mut sum_ixiy = 0.0f32;
                let mut sum_ixit = 0.0f32;
                let mut sum_iyit = 0.0f32;

                // Compute gradients in window
                for wy in y - half_win..=y + half_win {
                    for wx in x - half_win..=x + half_win {
                        if wy == 0
                            || wy >= height.saturating_sub(1)
                            || wx == 0
                            || wx >= width.saturating_sub(1)
                        {
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

        OpticalFlow::new(flow_x, flow_y)
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
    /// Validate a square convolution/morphology kernel against a frame.
    ///
    /// `kernel_size == 0` used to slip through and produce an entirely black
    /// (blur), entirely white (erosion) or NaN-scaled (box blur) output, and
    /// a kernel larger than the frame underflowed the
    /// `half_kernel..dim - half_kernel` loop bounds.
    fn validate_kernel(frame: &VideoFrame, kernel_size: usize) -> IoResult<()> {
        if kernel_size == 0 {
            return Err(IoError::ConfigError(
                "Filter kernel_size must be >= 1".into(),
            ));
        }
        let smallest = frame.width.min(frame.height);
        if kernel_size > smallest {
            return Err(IoError::ConfigError(format!(
                "Filter kernel_size {kernel_size} exceeds the frame's smallest dimension {smallest}"
            )));
        }
        Ok(())
    }

    /// Apply a filter to a video frame
    pub fn apply_filter(frame: &VideoFrame, filter: VideoFilter) -> IoResult<VideoFrame> {
        frame.validate()?;

        // Ensure grayscale for most operations
        let gray = if frame.channels != 1 {
            frame.to_grayscale()?
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
        Self::validate_kernel(frame, kernel_size)?;
        if !sigma.is_finite() || sigma <= 0.0 {
            // sigma == 0 makes the centre tap exp(0/0) = NaN, which poisons
            // the whole normalised kernel.
            return Err(IoError::ConfigError(format!(
                "Gaussian blur sigma must be finite and > 0, got {sigma}"
            )));
        }
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

        for y in half_kernel..height.saturating_sub(half_kernel) {
            for x in half_kernel..width.saturating_sub(half_kernel) {
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
        Self::validate_kernel(frame, kernel_size)?;
        let width = frame.width;
        let height = frame.height;
        let half_kernel = kernel_size / 2;
        let scale = 1.0 / (kernel_size * kernel_size) as f32;

        let mut result_data = vec![0u8; width * height];

        for y in half_kernel..height.saturating_sub(half_kernel) {
            for x in half_kernel..width.saturating_sub(half_kernel) {
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
        Self::validate_kernel(frame, 3)?;
        let width = frame.width;
        let height = frame.height;

        let sobel_x = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
        let sobel_y = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

        let mut result_data = vec![0u8; width * height];

        for y in 1..height.saturating_sub(1) {
            for x in 1..width.saturating_sub(1) {
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
        Self::validate_kernel(frame, 3)?;
        let width = frame.width;
        let height = frame.height;

        let laplacian = [0, -1, 0, -1, 4, -1, 0, -1, 0];

        let mut result_data = vec![0u8; width * height];

        for y in 1..height.saturating_sub(1) {
            for x in 1..width.saturating_sub(1) {
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
        Self::validate_kernel(frame, kernel_size)?;
        let width = frame.width;
        let height = frame.height;
        let half_kernel = kernel_size / 2;

        let mut result_data = vec![0u8; width * height];

        for y in half_kernel..height.saturating_sub(half_kernel) {
            for x in half_kernel..width.saturating_sub(half_kernel) {
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
        Self::validate_kernel(frame, kernel_size)?;
        let width = frame.width;
        let height = frame.height;
        let half_kernel = kernel_size / 2;

        let mut result_data = vec![0u8; width * height];

        for y in half_kernel..height.saturating_sub(half_kernel) {
            for x in half_kernel..width.saturating_sub(half_kernel) {
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
        Self::validate_kernel(frame, 3)?;
        let width = frame.width;
        let height = frame.height;

        let sharpen_kernel = [0, -1, 0, -1, 5, -1, 0, -1, 0];

        let mut result_data = vec![0u8; width * height];

        for y in 1..height.saturating_sub(1) {
            for x in 1..width.saturating_sub(1) {
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

#[cfg(test)]
mod tests {
    use super::*;

    // ================================================================
    // Robustness: user-controlled frame data must never panic (id=47)
    // ================================================================

    fn gray_frame(width: usize, height: usize) -> VideoFrame {
        VideoFrame {
            index: 0,
            timestamp: 0.0,
            width,
            height,
            channels: 1,
            data: (0..width * height).map(|i| (i % 251) as u8).collect(),
        }
    }

    #[test]
    fn test_optical_flow_on_degenerate_frames_does_not_underflow() {
        // 0x0, 1x1 and 2x2 frames all used to wrap `height - 1` /
        // `height - half_win` around to usize::MAX and index out of bounds.
        for (w, h) in [(0usize, 0usize), (1, 1), (2, 2), (3, 3), (4, 2)] {
            let prev = gray_frame(w, h);
            let curr = gray_frame(w, h);
            for method in [
                OpticalFlowMethod::DenseGradient,
                OpticalFlowMethod::LucasKanade,
                OpticalFlowMethod::BlockMatching,
            ] {
                let flow = OpticalFlowEstimator::new(method)
                    .compute(&prev, &curr)
                    .unwrap_or_else(|e| panic!("{method:?} on {w}x{h} failed: {e}"));
                assert_eq!(flow.dimensions(), (h, w));
            }
        }
    }

    #[test]
    fn test_optical_flow_with_zero_window_size_does_not_panic() {
        // `step_by(0)` in block matching used to panic.
        let prev = gray_frame(8, 8);
        let curr = gray_frame(8, 8);
        let estimator = OpticalFlowEstimator::new(OpticalFlowMethod::BlockMatching)
            .with_window_size(0)
            .with_pyramid_levels(1);
        assert!(estimator.compute(&prev, &curr).is_ok());
    }

    #[test]
    fn test_optical_flow_new_rejects_mismatched_shapes() {
        let flow_x = Array2::zeros((4, 4));
        let flow_y = Array2::zeros((2, 2));
        assert!(OpticalFlow::new(flow_x, flow_y).is_err());
    }

    #[test]
    fn test_filters_reject_degenerate_kernels() {
        let frame = gray_frame(8, 8);

        assert!(
            VideoProcessor::apply_filter(&frame, VideoFilter::BoxBlur { kernel_size: 0 }).is_err()
        );
        assert!(
            VideoProcessor::apply_filter(&frame, VideoFilter::Erosion { kernel_size: 0 }).is_err()
        );
        assert!(
            VideoProcessor::apply_filter(&frame, VideoFilter::Dilation { kernel_size: 32 })
                .is_err()
        );
        assert!(VideoProcessor::apply_filter(
            &frame,
            VideoFilter::GaussianBlur {
                kernel_size: 3,
                sigma: 0.0,
            }
        )
        .is_err());
        // A frame smaller than a 3x3 kernel is rejected rather than
        // underflowing the loop bounds.
        assert!(VideoProcessor::apply_filter(&gray_frame(2, 2), VideoFilter::SobelEdge).is_err());

        // Valid configurations still work.
        assert!(VideoProcessor::apply_filter(
            &frame,
            VideoFilter::GaussianBlur {
                kernel_size: 3,
                sigma: 1.0,
            }
        )
        .is_ok());
    }
}
