//! FFmpeg-backed `FfmpegReader`: the demuxer/decoder/scaler pipeline that
//! actually opens files, network streams and camera devices.
//!
//! This is a split of the former `video.rs`; see the parent `video` module
//! for the module-level documentation. `FfmpegReader` is one of the
//! backends behind the public [`super::VideoReader`] facade -- it is not
//! itself part of the crate's public API.

use super::types::{
    CameraDevice, FrameBuffer, PixelFormat, VideoConfig, VideoFrame, VideoMetadata, VideoSource,
};
use crate::error::{IoError, IoResult};
use ffmpeg_next as ffmpeg;
use tracing::{debug, info};

/// FFmpeg-backed video reader for streaming video frames.
///
/// Wraps a real FFmpeg demuxer + decoder + scaler: `new` opens the source
/// and fails if it cannot be opened or contains no video stream,
/// `read_frame` decodes actual frames, and `metadata` reports the
/// container's own fps / duration / dimensions.
///
/// A previous version never opened anything: `new()` returned `Ok` after only
/// calling `ffmpeg::init()`, `read_frame()` always returned `Ok(None)` (immediate
/// end-of-stream against a perfectly valid file) and `metadata()` reported a
/// hardcoded 30 fps, 0 s duration and 1920x1080 for every source.
///
/// Selected (or not) by the parent module's `resolve_backend` and driven
/// through the public [`super::VideoReader`] facade -- see that type for
/// the stable API.
pub(super) struct FfmpegReader {
    config: VideoConfig,
    buffer: FrameBuffer,
    input: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    scaler: Option<ffmpeg::software::scaling::Context>,
    stream_index: usize,
    /// Seconds per stream timestamp tick
    time_base: f64,
    metadata: VideoMetadata,
    /// Frames delivered to the caller so far
    current_frame: u64,
    /// Presentation timestamp (seconds) of the most recently delivered frame
    current_time: f64,
    /// Raw frames pulled out of the decoder (before decimation)
    decoded_frames: u64,
    /// The demuxer has been fully drained
    eof: bool,
}

impl FfmpegReader {
    /// Create a new video reader and open the configured source.
    ///
    /// # Errors
    ///
    /// * [`IoError::ReadFailed`] if the source cannot be opened.
    /// * [`IoError::Unsupported`] if it contains no decodable video stream, or
    ///   for a camera device whose capture backend FFmpeg does not provide.
    pub(super) async fn new(config: VideoConfig) -> IoResult<Self> {
        // Initialize FFmpeg
        ffmpeg::init()
            .map_err(|e| IoError::Connection(format!("Failed to initialize FFmpeg: {}", e)))?;

        let mut input = Self::open_source(&config)?;

        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| {
                IoError::Unsupported(format!(
                    "No video stream found in {:?}",
                    Self::source_label(&config.source)
                ))
            })?;

        let stream_index = stream.index();
        let time_base = f64::from(stream.time_base());
        let avg_rate = stream.avg_frame_rate();
        let fps = if avg_rate.denominator() != 0 {
            f64::from(avg_rate.numerator()) / f64::from(avg_rate.denominator())
        } else {
            0.0
        };
        // Containers such as Y4M carry neither a frame count nor a duration;
        // 0 then honestly means "unknown" rather than a fabricated value.
        let container_frames = stream.frames().max(0) as u64;
        let stream_duration = stream.duration();
        let parameters = stream.parameters();
        // `stream` borrows `input`; end the borrow before `input` is used again.
        let _ = stream;

        let duration = if stream_duration > 0 {
            stream_duration as f64 * time_base
        } else {
            let container = input.duration();
            if container > 0 {
                container as f64 / AV_TIME_BASE_F64
            } else {
                0.0
            }
        };

        let codec_context = ffmpeg::codec::context::Context::from_parameters(parameters)
            .map_err(|e| IoError::Unsupported(format!("Unsupported video codec: {}", e)))?;
        let decoder = codec_context
            .decoder()
            .video()
            .map_err(|e| IoError::Unsupported(format!("No video decoder available: {}", e)))?;

        let width = config
            .target_width
            .unwrap_or_else(|| decoder.width() as usize);
        let height = config
            .target_height
            .unwrap_or_else(|| decoder.height() as usize);
        if width == 0 || height == 0 {
            return Err(IoError::ConfigError(
                "Video frame dimensions must be non-zero".into(),
            ));
        }

        let metadata = VideoMetadata {
            fps,
            duration,
            frame_count: container_frames,
            width,
            height,
            pixel_format: config.pixel_format,
        };

        // Honour `start_time` by seeking before the first decode.
        if let Some(start) = config.start_time {
            if !start.is_finite() || start < 0.0 {
                return Err(IoError::ConfigError(format!(
                    "start_time must be finite and >= 0, got {start}"
                )));
            }
            let ts = (start * AV_TIME_BASE_F64) as i64;
            input
                .seek(ts, ..ts)
                .map_err(|e| IoError::ReadFailed(format!("Seek to {start}s failed: {e}")))?;
        }

        // A zero-capacity FrameBuffer can never accept a frame, which would
        // spin `read_frame` forever; clamp to at least one.
        let buffer = FrameBuffer::new(config.buffer_size.max(1));

        info!(
            backend = "ffmpeg",
            "VideoReader opened {}: {}x{}, {:.3} fps",
            Self::source_label(&config.source),
            width,
            height,
            fps
        );

        Ok(Self {
            config,
            buffer,
            input,
            decoder,
            scaler: None,
            stream_index,
            time_base,
            metadata,
            current_frame: 0,
            current_time: 0.0,
            decoded_frames: 0,
            eof: false,
        })
    }

    /// Human-readable description of a source, for error messages.
    fn source_label(source: &VideoSource) -> String {
        match source {
            VideoSource::File(path) => path.display().to_string(),
            VideoSource::Camera(device) => format!("camera {device}"),
            VideoSource::Network(url) => url.clone(),
        }
    }

    /// Open the configured source as an FFmpeg demuxer.
    fn open_source(config: &VideoConfig) -> IoResult<ffmpeg::format::context::Input> {
        match &config.source {
            VideoSource::File(path) => {
                if !path.exists() {
                    return Err(IoError::ReadFailed(format!(
                        "Video file not found: {}",
                        path.display()
                    )));
                }
                ffmpeg::format::input(path).map_err(|e| {
                    IoError::ReadFailed(format!("Failed to open {}: {}", path.display(), e))
                })
            }
            VideoSource::Network(url) => ffmpeg::format::input(std::path::Path::new(url))
                .map_err(|e| IoError::Connection(format!("Failed to open {url}: {e}"))),
            VideoSource::Camera(device) => Self::open_camera(config, device),
        }
    }

    /// Open a capture device (v4l2 / dshow / avfoundation) through FFmpeg's
    /// `libavdevice`, honouring `camera_format`, `camera_fps` and the target
    /// resolution.
    fn open_camera(config: &VideoConfig, device: &str) -> IoResult<ffmpeg::format::context::Input> {
        ffmpeg::device::register_all();

        let format_name = config
            .camera_format
            .clone()
            .unwrap_or_else(|| CameraDevice::default_format().to_string());
        let name = std::ffi::CString::new(format_name.as_str()).map_err(|_| {
            IoError::ConfigError(format!("Invalid camera format name: {format_name}"))
        })?;

        // SAFETY: `av_find_input_format` only reads the NUL-terminated string
        // we pass and returns either NULL or a pointer to one of libavdevice's
        // statically allocated `AVInputFormat` descriptors, which outlives the
        // process. The null case is handled below.
        let raw = unsafe { ffmpeg::ffi::av_find_input_format(name.as_ptr()) };
        if raw.is_null() {
            return Err(IoError::Unsupported(format!(
                "FFmpeg has no input device named '{format_name}' \
                 (this build of libavdevice does not provide it)"
            )));
        }
        // SAFETY: `raw` is a non-null pointer to a static AVInputFormat, as
        // required by `Input::wrap`. The `cast_mut` is sound: since FFmpeg 5
        // input-format descriptors are immutable static tables and
        // `av_find_input_format` returns `*const AVInputFormat`, while
        // ffmpeg-next 8.x's `Input::wrap` still takes `*mut` for historical
        // reasons. `Input` never writes through the pointer on our path --
        // every accessor goes through `as_ptr()` (const), and `open_with`
        // passes it back to `avformat_open_input`, whose parameter is
        // `const AVInputFormat *fmt` since FFmpeg 5. We never call
        // `as_mut_ptr()`.
        let format = ffmpeg::Format::Input(unsafe { ffmpeg::format::Input::wrap(raw.cast_mut()) });

        let mut options = ffmpeg::Dictionary::new();
        let fps_string;
        if let Some(fps) = config.camera_fps {
            fps_string = fps.to_string();
            options.set("framerate", &fps_string);
        }
        let size_string;
        if let (Some(width), Some(height)) = (config.target_width, config.target_height) {
            size_string = format!("{width}x{height}");
            options.set("video_size", &size_string);
        }

        match ffmpeg::format::open_with(std::path::Path::new(device), &format, options) {
            Ok(ffmpeg::format::Context::Input(input)) => Ok(input),
            Ok(_) => Err(IoError::Unsupported(format!(
                "Camera device '{device}' opened as an output context"
            ))),
            Err(e) => Err(IoError::Connection(format!(
                "Failed to open camera '{device}' via '{format_name}': {e}"
            ))),
        }
    }

    /// Read next frame.
    ///
    /// Returns `Ok(None)` only at genuine end-of-stream (or once `max_frames`
    /// has been delivered).
    pub(super) async fn read_frame(&mut self) -> IoResult<Option<VideoFrame>> {
        // Check if we've reached max frames
        if let Some(max) = self.config.max_frames {
            if self.current_frame >= max {
                return Ok(None);
            }
        }

        while self.buffer.is_empty() {
            if self.eof {
                // The demuxer is drained, but the decoder may still hold
                // reordered pictures that did not fit in the frame buffer on
                // the previous pass. Keep pulling until it genuinely yields
                // nothing, otherwise a small `buffer_size` would silently
                // truncate the tail of a B-frame stream.
                self.drain_decoder()?;
                if self.buffer.is_empty() {
                    break;
                }
            } else {
                self.pump_once()?;
            }
        }

        match self.buffer.pop() {
            Some(frame) => {
                self.current_frame += 1;
                self.current_time = frame.timestamp;
                Ok(Some(frame))
            }
            None => Ok(None),
        }
    }

    /// Feed one demuxed packet (or the end-of-file marker) into the decoder
    /// and move every frame it produces into the frame buffer.
    fn pump_once(&mut self) -> IoResult<()> {
        // Scoped so the `&mut self.input` borrow held by `packets()` ends
        // before the decoder (another field of `self`) is used.
        let next = {
            let mut packets = self.input.packets();
            packets
                .next()
                .map(|(stream, packet)| (stream.index(), packet))
        };

        match next {
            Some((index, packet)) => {
                if index == self.stream_index {
                    self.decoder.send_packet(&packet).map_err(|e| {
                        IoError::ReadFailed(format!("Decoder rejected packet: {e}"))
                    })?;
                    self.drain_decoder()?;
                }
            }
            None => {
                self.eof = true;
                // Flush the decoder so buffered (e.g. B-frame reordered)
                // pictures are still delivered.
                self.decoder
                    .send_eof()
                    .map_err(|e| IoError::ReadFailed(format!("Decoder flush failed: {e}")))?;
                self.drain_decoder()?;
            }
        }

        Ok(())
    }

    /// Pull every decoded picture currently available, applying decimation.
    fn drain_decoder(&mut self) -> IoResult<()> {
        let mut decoded = ffmpeg::util::frame::Video::empty();
        let decimation = self.config.decimation.max(1) as u64;

        while !self.buffer.is_full() {
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {}
                // EOF and EAGAIN both mean "no further picture right now";
                // anything else is a genuine decode failure and must not be
                // presented to the caller as end-of-stream.
                Err(ffmpeg::Error::Eof) => break,
                Err(ffmpeg::Error::Other { errno }) if errno == EAGAIN => break,
                Err(e) => return Err(IoError::ReadFailed(format!("Decoding failed: {e}"))),
            }

            let raw_index = self.decoded_frames;
            self.decoded_frames += 1;
            if !raw_index.is_multiple_of(decimation) {
                continue;
            }

            let frame = self.convert_frame(&decoded, raw_index)?;
            if !self.buffer.push(frame) {
                break;
            }
        }

        Ok(())
    }

    /// Scale/convert a decoded picture into the configured output format.
    fn convert_frame(
        &mut self,
        decoded: &ffmpeg::util::frame::Video,
        raw_index: u64,
    ) -> IoResult<VideoFrame> {
        let dst_format = ffmpeg_pixel(self.config.pixel_format);
        let dst_width = self
            .config
            .target_width
            .unwrap_or_else(|| decoded.width() as usize) as u32;
        let dst_height = self
            .config
            .target_height
            .unwrap_or_else(|| decoded.height() as usize) as u32;
        if dst_width == 0 || dst_height == 0 {
            return Err(IoError::ConfigError(
                "Video frame dimensions must be non-zero".into(),
            ));
        }

        let scaler = match self.scaler {
            Some(ref mut scaler) => scaler,
            None => {
                let created = ffmpeg::software::scaling::Context::get(
                    decoded.format(),
                    decoded.width(),
                    decoded.height(),
                    dst_format,
                    dst_width,
                    dst_height,
                    ffmpeg::software::scaling::Flags::BILINEAR,
                )
                .map_err(|e| IoError::Unsupported(format!("Cannot build pixel converter: {e}")))?;
                self.scaler.insert(created)
            }
        };

        let mut converted = ffmpeg::util::frame::Video::empty();
        scaler
            .run(decoded, &mut converted)
            .map_err(|e| IoError::ReadFailed(format!("Pixel conversion failed: {e}")))?;

        let channels = self.config.pixel_format.channels();
        let width = converted.width() as usize;
        let height = converted.height() as usize;
        let stride = converted.stride(0);
        let row_bytes = width * channels;
        let plane = converted.data(0);

        let mut data = Vec::with_capacity(row_bytes * height);
        for row in 0..height {
            let start = row * stride;
            let end = start + row_bytes;
            let slice = plane
                .get(start..end)
                .ok_or_else(|| IoError::ReadFailed("Truncated frame plane".to_string()))?;
            data.extend_from_slice(slice);
        }

        let timestamp = decoded
            .timestamp()
            .map(|pts| pts as f64 * self.time_base)
            .unwrap_or(0.0);

        Ok(VideoFrame {
            index: raw_index,
            timestamp,
            width,
            height,
            channels,
            data,
        })
    }

    /// Get video metadata, read from the container/decoder.
    ///
    /// `frame_count` and `duration` are `0` when the container does not report
    /// them (raw formats such as Y4M) -- that is an honest "unknown", not a
    /// guess.
    pub(super) fn metadata(&self) -> VideoMetadata {
        self.metadata.clone()
    }

    /// Seek to a specific time (seconds) in the source.
    pub(super) async fn seek(&mut self, time: f64) -> IoResult<()> {
        if !time.is_finite() || time < 0.0 {
            return Err(IoError::ConfigError(format!(
                "Seek target must be finite and >= 0, got {time}"
            )));
        }

        debug!("Seeking video source to {time}s");

        let ts = (time * AV_TIME_BASE_F64) as i64;
        self.input
            .seek(ts, ..ts)
            .map_err(|e| IoError::ReadFailed(format!("Seek to {time}s failed: {e}")))?;

        self.decoder.flush();
        self.buffer.clear();
        self.eof = false;
        self.current_time = time;
        Ok(())
    }

    /// Number of frames delivered by `read_frame` so far
    pub(super) fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Presentation timestamp (seconds) of the most recently delivered frame
    pub(super) fn current_time(&self) -> f64 {
        self.current_time
    }
}

/// FFmpeg's internal timestamp resolution (`AV_TIME_BASE`), in ticks/second.
const AV_TIME_BASE_F64: f64 = 1_000_000.0;

/// POSIX `EAGAIN` for the target platform (11 on Linux, 35 on the BSDs and
/// macOS), as wrapped by `ffmpeg::Error::Other`. FFmpeg returns
/// `AVERROR(EAGAIN)` from `avcodec_receive_frame` to mean "feed me another
/// packet", which is not an error condition for a streaming reader.
const EAGAIN: i32 = ffmpeg::ffi::EAGAIN;

/// Map a [`PixelFormat`] onto the corresponding FFmpeg pixel format.
fn ffmpeg_pixel(format: PixelFormat) -> ffmpeg::format::Pixel {
    match format {
        PixelFormat::Rgb => ffmpeg::format::Pixel::RGB24,
        PixelFormat::Gray => ffmpeg::format::Pixel::GRAY8,
        PixelFormat::Rgba => ffmpeg::format::Pixel::RGBA,
    }
}
