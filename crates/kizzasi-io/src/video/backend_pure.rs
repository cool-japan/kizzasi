//! Pure-Rust video backend (the OxiMedia stack): `PureReader`.
//!
//! Decodes Y4M (YUV4MPEG2) files and captures from a camera without linking a
//! single line of C: `oximedia-container`'s `Y4mDemuxer` splits a file's byte
//! stream into raw planar pictures, `oximedia-capture` drives the platform
//! capture API for a live device, `oximedia-core`'s `convert::pixel` maps the
//! samples onto the configured [`PixelFormat`], and `oximedia-cv`'s bilinear
//! resampler rescales to `target_width`/`target_height`.
//!
//! `PureReader` is one of the backends behind the public
//! [`super::VideoReader`] facade -- it is not itself part of the crate's
//! public API. Its observable behaviour is deliberately identical to
//! `backend_ffmpeg`'s `FfmpegReader` for the same source: the same decimation
//! and buffering sequence, the same `max_frames` accounting, the same raw
//! (pre-decimation) [`VideoFrame::index`], the same timestamps, and the same
//! honest `frame_count`/`duration` of `0` for a source that reports neither.
//! The reader tests in `reader_tests` run against every compiled-in backend
//! with identical assertions, so that parity is enforced rather than hoped
//! for.
//!
//! ## Module split
//!
//! This file owns the Y4M reader, the [`PureReader`] shell shared by every
//! source, and the conversion/rescale helpers both sources use. The camera
//! path -- opening a device, the `camera_format` alias table, `CaptureError`
//! mapping and the capture pixel-layout conversions -- lives in the sibling
//! [`super::backend_pure_camera`] module, which is a size split rather than a
//! layering one: it reaches back into the `pub(super)` helpers here rather
//! than duplicating them.
//!
//! ## What this backend does *not* do yet
//!
//! * **Network streams** -- no pure-Rust RTSP/HTTP demuxer is wired up.
//! * **Compressed containers** (MP4/MKV/WebM/...) -- `oximedia-container`
//!   has demuxers for several of them, but no decoders are hooked up here,
//!   so claiming support would mean claiming to decode H.264.
//! * **Y4M `C444alpha`** -- rejected at open time (see [`PureReader::new`]).
//! * **Camera capture on a platform `oximedia-capture` has no backend for.**
//!   [`PureReader::supports`] answers `false` for [`VideoSource::Camera`]
//!   there, so `Auto` keeps routing cameras to FFmpeg rather than claiming a
//!   source it would then fail to open.
//!
//! Every one of those returns an [`IoError::Unsupported`] naming what is
//! missing, rather than an `Ok` that quietly decodes nothing.

use super::backend_pure_camera::CameraSource;
use super::types::{FrameBuffer, PixelFormat, VideoConfig, VideoFrame, VideoMetadata, VideoSource};
use crate::error::{IoError, IoResult};
use oximedia_container::demux::y4m::{Y4mChroma, Y4mDemuxer};
use oximedia_core::convert::pixel::{
    gray8_to_rgb24, yuv420p_to_gray8, yuv420p_to_rgb24, ColorMatrix, PixelConverter,
};
use oximedia_cv::image::resize::{resize_image, ResizeConfig, ResizeMethod};
use std::fs::File;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Pure-Rust video reader.
///
/// Mirrors `FfmpegReader`'s field set (and its meaning) one for one, because
/// the two are held to the same observable contract -- only `source`
/// differs, holding an OxiMedia demuxer instead of an FFmpeg
/// demuxer/decoder/scaler triple.
pub(super) struct PureReader {
    config: VideoConfig,
    buffer: FrameBuffer,
    source: PureSource,
    metadata: VideoMetadata,
    /// Frames delivered to the caller so far
    current_frame: u64,
    /// Presentation timestamp (seconds) of the most recently delivered frame
    current_time: f64,
    /// Raw frames pulled out of the demuxer (before decimation)
    decoded_frames: u64,
    /// The demuxer has been fully drained
    eof: bool,
}

/// The opened source behind a [`PureReader`].
///
/// Both variants carry their bulk behind a `Box`: a [`CameraSource`] is a
/// whole capture session (device handle, join handle, channel endpoints,
/// statistics) and a `Y4mDemuxer` is ~200 bytes of reader and cached header
/// state. Inlining either would pad every reader of the *other* kind out to
/// its size, which is exactly what `clippy::large_enum_variant` objects to.
enum PureSource {
    /// A Y4M (YUV4MPEG2) file.
    Y4m {
        /// Kept so [`PureReader::seek`] can reopen the file from the start.
        path: PathBuf,
        /// `Y4mDemuxer` wraps whatever `Read` it is given in a `BufReader`
        /// of its own, so handing it the bare `File` (rather than a
        /// `BufReader<File>`) avoids stacking two buffers and copying every
        /// byte twice.
        ///
        /// Boxed for the same reason [`PureSource::Camera`] is: the demuxer
        /// is ~200 bytes of cached header and reader state, and inlining it
        /// would pad every camera reader out to that size for nothing. One
        /// allocation per opened file, against a per-frame decode.
        demux: Box<Y4mDemuxer<File>>,
        /// Source picture geometry and chroma-plane shape, cached from the
        /// header so no per-frame reparse is needed.
        geometry: PlaneGeometry,
        /// Header chroma mode, cached alongside `geometry`.
        chroma: Y4mChroma,
        /// `fps_den / fps_num`: the presentation-timestamp step between two
        /// consecutive raw frames, in seconds.
        seconds_per_frame: f64,
    },
    /// A live capture device.
    Camera(Box<CameraSource>),
}

impl PureReader {
    /// Whether the pure backend can currently handle `source`.
    ///
    /// True for a [`VideoSource::File`] whose first bytes carry the
    /// `YUV4MPEG2` magic, and for a [`VideoSource::Camera`] on a platform
    /// `oximedia-capture` actually has a backend for. `Auto` backend
    /// resolution consults this before preferring the pure backend over
    /// FFmpeg, so a build with both features gets pure-Rust Y4M decoding and
    /// pure-Rust capture, and FFmpeg for everything else.
    ///
    /// The camera answer is deliberately *not* an unconditional `true`.
    /// `oximedia-capture` reports [`BackendKind::Unsupported`][bk] on a
    /// target whose capture API it has not implemented, and claiming the
    /// source there would take cameras away from FFmpeg (which can still open
    /// them) only to fail at open time -- the opposite of `Auto`'s "error
    /// only if neither backend can handle the source" rule.
    ///
    /// Network streams have no pure-Rust demuxer wired up at all.
    ///
    /// [bk]: oximedia_capture::BackendKind::Unsupported
    pub(super) fn supports(source: &VideoSource) -> bool {
        match source {
            // A probe failure (missing file) is not this function's error to
            // report -- `resolve_auto` runs the same probe and surfaces it.
            VideoSource::File(path) => super::probe_is_y4m(path).unwrap_or(false),
            VideoSource::Camera(_) => super::backend_pure_camera::host_capture_is_available(),
            VideoSource::Network(_) => false,
        }
    }

    /// Open the configured source with the pure-Rust stack.
    ///
    /// # Errors
    ///
    /// * [`IoError::ReadFailed`] if the source is a file that does not exist
    ///   (same variant and wording the FFmpeg backend uses), or cannot be
    ///   opened / parsed as Y4M.
    /// * [`IoError::Unsupported`] for anything this backend does not decode
    ///   yet -- a non-Y4M file, a network stream, a Y4M stream in the one
    ///   chroma mode (`C444alpha`) it has no output format for, or a camera
    ///   whose `camera_format` names another platform's capture API.
    /// * [`IoError::ConfigError`] for a nonsensical `start_time` or zero
    ///   output dimensions.
    /// * For a camera, whatever `backend_pure_camera::map_capture_error`
    ///   makes of an `oximedia-capture` open failure -- typically
    ///   [`IoError::Connection`] (no such device, permission refused) or
    ///   [`IoError::ConfigError`] (no mode satisfies the request).
    pub(super) async fn new(config: VideoConfig) -> IoResult<Self> {
        let path = match &config.source {
            VideoSource::File(path) => path.clone(),
            VideoSource::Camera(device) => {
                let camera = CameraSource::open(&config, device)?;
                return Self::from_camera_source(config, camera);
            }
            VideoSource::Network(url) => {
                return Err(IoError::Unsupported(format!(
                    "the pure-Rust video backend cannot open network stream '{url}': it \
                     decodes Y4M (YUV4MPEG2) files only -- enable the `video` (FFmpeg) \
                     feature for network sources"
                )))
            }
        };

        // Reuses the parent module's probe, which reports a missing file as
        // `IoError::ReadFailed("Video file not found: ...")` -- byte for byte
        // what `FfmpegReader::open_source` reports for the same path.
        // Forcing `VideoBackend::Pure` bypasses `resolve_auto` entirely, so
        // this check has to live here too rather than only there.
        if !super::probe_is_y4m(&path)? {
            return Err(IoError::Unsupported(format!(
                "the pure-Rust video backend cannot decode {}: it supports Y4M \
                 (YUV4MPEG2) files only today -- enable the `video` (FFmpeg) feature for \
                 other containers, or leave the backend on `Auto` to get that \
                 automatically",
                path.display()
            )));
        }

        let demux = open_demuxer(&path)?;
        let header = demux.header();
        let chroma = header.chroma;
        let source_width = header.width as usize;
        let source_height = header.height as usize;
        let fps_num = header.fps_num;
        let fps_den = header.fps_den;

        let geometry = PlaneGeometry::new(source_width, source_height, chroma)?;

        // `fps_den == 0` is already rejected by the Y4M header parser and
        // `fps_num == 0` means "unknown rate"; either way 0.0 is the honest
        // answer rather than an infinity that would poison every timestamp.
        let fps = if fps_den == 0 {
            0.0
        } else {
            f64::from(fps_num) / f64::from(fps_den)
        };
        let seconds_per_frame = if fps_num == 0 {
            0.0
        } else {
            f64::from(fps_den) / f64::from(fps_num)
        };

        let width = config.target_width.unwrap_or(source_width);
        let height = config.target_height.unwrap_or(source_height);
        if width == 0 || height == 0 {
            return Err(IoError::ConfigError(
                "Video frame dimensions must be non-zero".into(),
            ));
        }

        let metadata = VideoMetadata {
            fps,
            // A Y4M header carries neither a duration nor a frame count, so
            // 0 is an honest "unknown" -- and it is exactly what the FFmpeg
            // backend reports for the same file, where `stream.duration()`
            // and `stream.frames()` come back as 0 for this container.
            // Deriving them would mean reading the whole file at open time.
            duration: 0.0,
            frame_count: 0,
            width,
            height,
            pixel_format: config.pixel_format,
        };

        // Validated here, before anything is logged as "opened", so an
        // impossible `start_time` fails the same way (and at the same point)
        // as in the FFmpeg backend, which checks it just before its own
        // opening log line. The actual drain happens once `Self` exists.
        let start_time = match config.start_time {
            Some(start) if !start.is_finite() || start < 0.0 => {
                return Err(IoError::ConfigError(format!(
                    "start_time must be finite and >= 0, got {start}"
                )))
            }
            other => other,
        };

        // A zero-capacity FrameBuffer can never accept a frame, which would
        // spin `read_frame` forever; clamp to at least one.
        let buffer = FrameBuffer::new(config.buffer_size.max(1));

        info!(
            backend = "pure",
            "VideoReader opened {}: {}x{}, {:.3} fps (Y4M C{})",
            path.display(),
            width,
            height,
            fps,
            chroma
        );

        let mut reader = Self {
            config,
            buffer,
            source: PureSource::Y4m {
                path,
                demux,
                geometry,
                chroma,
                seconds_per_frame,
            },
            metadata,
            current_frame: 0,
            current_time: 0.0,
            decoded_frames: 0,
            eof: false,
        };

        // Honour `start_time` before the first read, the way the FFmpeg
        // backend seeks before its first decode.
        if let Some(start) = start_time {
            reader.skip_raw_frames(frames_before(start, fps))?;
            reader.current_time = reader.decoded_frames as f64 * seconds_per_frame;
        }

        Ok(reader)
    }

    /// Build a reader around an already-opened capture device.
    ///
    /// Shared by [`PureReader::new`]'s camera arm and the
    /// [`PureReader::from_capture_session`] test seam, so a scripted mock
    /// session and a real device travel identical code from here on.
    ///
    /// # Errors
    ///
    /// * [`IoError::Unsupported`] if `start_time` is set. A capture device
    ///   has no past to start from, and quietly discarding the first N
    ///   seconds of a live stream is not a seek -- it is data loss the caller
    ///   never asked for. Same wording as [`PureReader::seek`]'s refusal.
    /// * [`IoError::ConfigError`] for zero output dimensions.
    fn from_camera_source(config: VideoConfig, camera: CameraSource) -> IoResult<Self> {
        if config.start_time.is_some() {
            return Err(IoError::Unsupported(format!(
                "start_time is not supported on camera '{}': the pure-Rust backend cannot \
                 seek a live capture device, and skipping frames off the front of a live \
                 stream would silently discard them",
                camera.device()
            )));
        }

        let negotiated = camera.negotiated();
        let source_width = negotiated.width as usize;
        let source_height = negotiated.height as usize;
        let width = config.target_width.unwrap_or(source_width);
        let height = config.target_height.unwrap_or(source_height);
        if width == 0 || height == 0 {
            return Err(IoError::ConfigError(
                "Video frame dimensions must be non-zero".into(),
            ));
        }

        let fps = camera.fps();
        let metadata = VideoMetadata {
            fps,
            // A live device has neither: it runs until it is stopped. `0` is
            // the same honest "unknown" the Y4M path reports, and what the
            // FFmpeg backend reports for a capture input too.
            duration: 0.0,
            frame_count: 0,
            width,
            height,
            pixel_format: config.pixel_format,
        };

        // A zero-capacity FrameBuffer can never accept a frame, which would
        // spin `read_frame` forever; clamp to at least one.
        let buffer = FrameBuffer::new(config.buffer_size.max(1));

        info!(
            backend = "pure",
            capture = %camera.backend(),
            "VideoReader opened camera {}: {}x{}, {:.3} fps (capture {} at {}x{})",
            camera.device(),
            width,
            height,
            fps,
            negotiated.encoding,
            negotiated.width,
            negotiated.height
        );

        Ok(Self {
            config,
            buffer,
            source: PureSource::Camera(Box::new(camera)),
            metadata,
            current_frame: 0,
            current_time: 0.0,
            decoded_frames: 0,
            eof: false,
        })
    }

    /// Test seam: build a reader around a capture session the caller already
    /// opened.
    ///
    /// Exists so the camera path can be exercised without hardware:
    /// `oximedia-capture`'s `mock` backend is opened through
    /// `mock::open(CaptureConfig, MockScript)`, which cannot be reached
    /// through [`PureReader::new`] (that calls `oximedia_capture::open`, the
    /// *native* backend, by design -- a production path that could be
    /// redirected to a synthetic device would be a production path that can
    /// silently deliver synthetic frames). Everything after this point is the
    /// same code a real device runs.
    #[cfg(test)]
    pub(super) fn from_capture_session(
        config: VideoConfig,
        session: oximedia_capture::CaptureSession,
    ) -> IoResult<Self> {
        Self::from_camera_source(config, CameraSource::adopt(session)?)
    }

    /// Read the next frame.
    ///
    /// Returns `Ok(None)` only at genuine end-of-stream (or once
    /// `max_frames` has been delivered).
    pub(super) async fn read_frame(&mut self) -> IoResult<Option<VideoFrame>> {
        // `max_frames` counts *delivered* frames, not demuxed ones, exactly
        // as in the FFmpeg path.
        if let Some(max) = self.config.max_frames {
            if self.current_frame >= max {
                return Ok(None);
            }
        }

        // The FFmpeg path has a second, `eof`-only branch here that keeps
        // flushing its decoder after the demuxer runs dry, because a codec
        // with reordered pictures (B-frames) can still be holding output
        // that did not fit in the frame buffer. Y4M is raw, uncompressed and
        // already in presentation order: there is no decoder, so nothing can
        // be held back, and once the demuxer says end-of-stream there is
        // genuinely nothing left to drain.
        while self.buffer.is_empty() && !self.eof {
            self.pump_once()?;
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

    /// Fill the frame buffer from whichever source is open.
    fn pump_once(&mut self) -> IoResult<()> {
        match &self.source {
            PureSource::Y4m { .. } => self.pump_y4m(),
            PureSource::Camera(_) => self.pump_camera(),
        }
    }

    /// Pull raw pictures out of the demuxer until the frame buffer is full
    /// (or the stream ends), applying decimation and format conversion.
    ///
    /// The twin of `FfmpegReader::drain_decoder`, down to the
    /// `while !buffer.is_full()` bound: because `FrameBuffer` only frees
    /// space once it has been drained completely, that bound is what keeps a
    /// small `buffer_size` from dropping frames instead of merely
    /// backpressuring.
    fn pump_y4m(&mut self) -> IoResult<()> {
        let decimation = self.config.decimation.max(1) as u64;

        while !self.buffer.is_full() {
            let raw = match self.next_raw_frame()? {
                Some(raw) => raw,
                None => {
                    self.eof = true;
                    break;
                }
            };

            let raw_index = self.decoded_frames;
            self.decoded_frames += 1;
            if !raw_index.is_multiple_of(decimation) {
                continue;
            }

            let frame = self.convert_frame(&raw, raw_index)?;
            // Defensive: `VideoFrame`'s fields are public and every consumer
            // validates before indexing, so a conversion bug should surface
            // here as an error rather than downstream as a mismatch.
            frame.validate()?;
            if !self.buffer.push(frame) {
                break;
            }
        }

        Ok(())
    }

    /// Take **one** frame off the capture queue, applying decimation and
    /// format conversion.
    ///
    /// Deliberately not the `while !buffer.is_full()` loop `pump_y4m` uses.
    /// Batch-filling costs nothing against a file, but against a live device
    /// it would make the first `read_frame()` wait for `buffer_size` frame
    /// periods -- a third of a second at the default 5 frames / 30 fps, and
    /// twice that under `decimation = 2` -- for a buffer the caller is about
    /// to drain one frame at a time anyway. `read_frame`'s own
    /// `while buffer.is_empty() && !eof` loop already handles the case where
    /// this call decimates its frame away and pushes nothing, so the frame
    /// indices, the decimation sequence and the `max_frames` accounting are
    /// identical to the file path's; only the latency differs.
    fn pump_camera(&mut self) -> IoResult<()> {
        let decimation = self.config.decimation.max(1) as u64;
        let target = self.frame_target()?;

        let camera = match &mut self.source {
            PureSource::Camera(camera) => camera,
            // Unreachable: `pump_once` dispatches on the same variant. An
            // honest error rather than `unreachable!()`, so a future bug in
            // that dispatch fails loudly instead of aborting the process.
            PureSource::Y4m { .. } => {
                return Err(IoError::ReadFailed(
                    "internal: the camera pump was handed a Y4M source".to_string(),
                ))
            }
        };

        let frame = match camera.recv()? {
            Some(frame) => frame,
            None => {
                self.eof = true;
                return Ok(());
            }
        };

        let raw_index = self.decoded_frames;
        self.decoded_frames += 1;
        if !raw_index.is_multiple_of(decimation) {
            return Ok(());
        }

        let data = camera.convert(&frame, target)?;
        let converted = VideoFrame {
            index: raw_index,
            // The capture session's own timeline, normalised to start at zero
            // and guaranteed non-decreasing (a device clock that jumps
            // backwards is clamped upstream and counted). Derived from the
            // device where the backend can read its clock, from host arrival
            // time where it cannot -- either way it is measured, not
            // synthesised from a nominal frame rate the way Y4M's is.
            timestamp: frame.timestamp.as_secs_f64(),
            width: target.width,
            height: target.height,
            channels: target.format.channels(),
            data,
        };
        // Defensive: `VideoFrame`'s fields are public and every consumer
        // validates before indexing, so a conversion bug should surface here
        // as an error rather than downstream as a mismatch.
        converted.validate()?;

        if !self.buffer.push(converted) {
            // Unreachable: this runs only while the buffer is empty (see
            // `read_frame`) and its capacity is clamped to at least one.
            // Reported rather than swallowed, because swallowing it would
            // drop a captured frame silently.
            return Err(IoError::ReadFailed(
                "internal: the frame buffer refused a captured frame while it was empty"
                    .to_string(),
            ));
        }

        Ok(())
    }

    /// The output picture the current configuration asks for, given the
    /// source geometry.
    fn frame_target(&self) -> IoResult<FrameTarget> {
        let (source_width, source_height) = match &self.source {
            PureSource::Y4m { geometry, .. } => (geometry.width, geometry.height),
            PureSource::Camera(camera) => {
                let negotiated = camera.negotiated();
                (negotiated.width as usize, negotiated.height as usize)
            }
        };

        let target = FrameTarget::new(
            self.config.pixel_format,
            self.config.target_width.unwrap_or(source_width),
            self.config.target_height.unwrap_or(source_height),
        );
        if target.width == 0 || target.height == 0 {
            return Err(IoError::ConfigError(
                "Video frame dimensions must be non-zero".into(),
            ));
        }
        Ok(target)
    }

    /// Demux one raw picture, or `Ok(None)` at end-of-stream.
    fn next_raw_frame(&mut self) -> IoResult<Option<Vec<u8>>> {
        match &mut self.source {
            PureSource::Y4m { path, demux, .. } => demux.read_frame().map_err(|e| {
                IoError::ReadFailed(format!(
                    "Failed to read a frame from {}: {e}",
                    path.display()
                ))
            }),
            // Only reachable through `pump_y4m` / `skip_raw_frames`, and both
            // of those are gated on a Y4M source: `skip_raw_frames` runs for
            // `start_time` and `seek`, which `from_camera_source` and `seek`
            // refuse for a camera before they can get here.
            PureSource::Camera(camera) => Err(IoError::Unsupported(format!(
                "camera '{}' delivers frames through its capture queue, not a demuxer",
                camera.device()
            ))),
        }
    }

    /// Discard `count` raw pictures, advancing `decoded_frames` past them.
    ///
    /// O(n): the frames are demuxed and thrown away, because `Y4mDemuxer`
    /// only offers sequential `Read` access. An O(1) skip is possible in
    /// principle -- a Y4M frame occupies exactly `frame_size` bytes after
    /// its `FRAME` line, so a file whose frame tags carry no parameters has
    /// a constant stride that a single `Seek` could jump -- but that needs
    /// the demuxer to expose a `Seek`-bounded API (`Y4mDemuxer<R: Read +
    /// Seek>`), which is an upstream OxiMedia change rather than something
    /// to reimplement here against its internal buffer state.
    fn skip_raw_frames(&mut self, count: u64) -> IoResult<()> {
        for _ in 0..count {
            if self.next_raw_frame()?.is_none() {
                self.eof = true;
                break;
            }
            self.decoded_frames += 1;
        }
        Ok(())
    }

    /// Convert one raw planar picture into the configured output format.
    fn convert_frame(&self, raw: &[u8], raw_index: u64) -> IoResult<VideoFrame> {
        let (geometry, chroma, seconds_per_frame) = match &self.source {
            PureSource::Y4m {
                geometry,
                chroma,
                seconds_per_frame,
                ..
            } => (*geometry, *chroma, *seconds_per_frame),
            // Unreachable: only `pump_y4m` calls this, and it runs only for a
            // Y4M source. The camera path converts through
            // `backend_pure_camera::convert_capture_video_frame` instead,
            // because a capture frame arrives as typed planes rather than one
            // flat Y4M picture.
            PureSource::Camera(_) => {
                return Err(IoError::ReadFailed(
                    "internal: the Y4M converter was handed a capture source".to_string(),
                ))
            }
        };

        let target = self.frame_target()?;

        let data = convert_y4m_frame(raw, chroma, geometry, target)?;

        Ok(VideoFrame {
            index: raw_index,
            // Y4M has no per-frame timestamps: presentation time is implied
            // by the header frame rate, which is exactly how FFmpeg's
            // `yuv4mpegpipe` demuxer synthesises the PTS this mirrors
            // (`pts * time_base`, `time_base = fps_den/fps_num`). Derived
            // from the raw index so it keeps increasing under decimation.
            timestamp: raw_index as f64 * seconds_per_frame,
            width: target.width,
            height: target.height,
            channels: target.format.channels(),
            data,
        })
    }

    /// Get video metadata, read from the Y4M header.
    ///
    /// `frame_count` and `duration` are `0` because the container reports
    /// neither -- an honest "unknown", matching what the FFmpeg backend
    /// reports for the same file.
    pub(super) fn metadata(&self) -> VideoMetadata {
        self.metadata.clone()
    }

    /// Seek to a specific time (seconds) in the source.
    ///
    /// Reopens the file and re-drains from the start: see
    /// [`PureReader::skip_raw_frames`] for why this is O(n) and what the
    /// upstream O(1) alternative would be.
    ///
    /// # Errors
    ///
    /// [`IoError::Unsupported`] for a camera. A live capture device has no
    /// stored past to rewind into and no future to skip to; the only thing
    /// that could be implemented under this name is "throw frames away until
    /// the timestamp passes", which is not a seek and would block for as long
    /// as the requested offset.
    pub(super) async fn seek(&mut self, time: f64) -> IoResult<()> {
        if let PureSource::Camera(camera) = &self.source {
            return Err(IoError::Unsupported(format!(
                "cannot seek a live capture device (camera '{}')",
                camera.device()
            )));
        }

        if !time.is_finite() || time < 0.0 {
            return Err(IoError::ConfigError(format!(
                "Seek target must be finite and >= 0, got {time}"
            )));
        }

        debug!("Seeking video source to {time}s (pure backend, reopen + drain)");

        self.reopen()?;
        self.buffer.clear();
        self.eof = false;
        self.decoded_frames = 0;
        self.skip_raw_frames(frames_before(time, self.metadata.fps))?;

        let seconds_per_frame = match &self.source {
            PureSource::Y4m {
                seconds_per_frame, ..
            } => *seconds_per_frame,
            // Unreachable: the camera case returned at the top of `seek`.
            PureSource::Camera(_) => 0.0,
        };
        // The frame-quantised position actually reached, which for a
        // frame-aligned drain is the honest number. (`FfmpegReader::seek`
        // stores the requested `time` verbatim instead; its seek lands on a
        // keyframe, so neither value is the exact PTS of the next frame and
        // no test pins either one.)
        self.current_time = self.decoded_frames as f64 * seconds_per_frame;

        Ok(())
    }

    /// Reopen the underlying file and start a fresh demuxer at frame 0.
    ///
    /// The cached header fields (`geometry`, `chroma`, `seconds_per_frame`)
    /// are left alone: reparsing the same file yields the same values.
    fn reopen(&mut self) -> IoResult<()> {
        match &mut self.source {
            PureSource::Y4m { path, demux, .. } => {
                *demux = open_demuxer(path)?;
            }
            // Unreachable: only `seek` calls this, and it refuses a camera
            // before it gets here. Reopening a capture device would restart
            // the negotiation and the capture thread -- a different session,
            // not a reposition.
            PureSource::Camera(camera) => {
                return Err(IoError::Unsupported(format!(
                    "cannot reopen a live capture device (camera '{}')",
                    camera.device()
                )))
            }
        }
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

/// Open `path` and parse its Y4M header.
fn open_demuxer(path: &Path) -> IoResult<Box<Y4mDemuxer<File>>> {
    let file = File::open(path)
        .map_err(|e| IoError::ReadFailed(format!("Failed to open {}: {}", path.display(), e)))?;
    Y4mDemuxer::new(file).map(Box::new).map_err(|e| {
        IoError::ReadFailed(format!(
            "Failed to parse the Y4M header of {}: {}",
            path.display(),
            e
        ))
    })
}

/// Number of whole frames that precede `time` at `fps`.
///
/// Saturates rather than wrapping for absurd inputs; a count past the end of
/// the stream simply drains to end-of-stream.
fn frames_before(time: f64, fps: f64) -> u64 {
    let frames = (time * fps).round();
    if frames.is_finite() && frames > 0.0 {
        frames as u64
    } else {
        0
    }
}

/// Geometry of one raw picture: luma dimensions plus the chroma subsampling
/// of the layout it came in.
///
/// Named for Y4M, where it started, but the capture path in
/// [`super::backend_pure_camera`] builds one too -- the shifts describe any
/// planar layout, not just a Y4M chroma mode.
#[derive(Clone, Copy)]
pub(super) struct PlaneGeometry {
    /// Luma plane width in pixels.
    width: usize,
    /// Luma plane height in pixels.
    height: usize,
    /// log2 of the horizontal chroma subsampling factor: 1 for 4:2:0 and
    /// 4:2:2, 0 for 4:4:4 and monochrome.
    x_shift: u32,
    /// log2 of the vertical chroma subsampling factor: 1 for 4:2:0, 0
    /// otherwise.
    y_shift: u32,
}

impl PlaneGeometry {
    /// Derive the plane layout for `chroma` at `width` x `height`.
    ///
    /// # Errors
    ///
    /// [`IoError::Unsupported`] for `C444alpha`: a Y4M alpha plane has no
    /// destination in any [`PixelFormat`] this crate offers -- `Rgba`'s
    /// alpha is a fixed 255 -- so silently dropping it would be a lie about
    /// what was decoded. Rejecting at open time (rather than on the first
    /// frame) mirrors FFmpeg failing at decoder construction.
    fn new(width: usize, height: usize, chroma: Y4mChroma) -> IoResult<Self> {
        let (x_shift, y_shift) = match chroma {
            Y4mChroma::C420jpeg | Y4mChroma::C420mpeg2 | Y4mChroma::C420paldv => (1, 1),
            Y4mChroma::C422 => (1, 0),
            // Monochrome has no chroma planes at all; the shifts are unused
            // for it (see `convert_mono`), so 0 is simply the neutral value.
            Y4mChroma::C444 | Y4mChroma::Mono => (0, 0),
            Y4mChroma::C444alpha => {
                return Err(IoError::Unsupported(
                    "Y4M C444alpha is not supported by the pure-Rust video backend: its \
                     alpha plane has no destination in PixelFormat::{Rgb,Rgba,Gray} \
                     (Rgba's alpha is a constant 255), and dropping it silently would \
                     misreport what was decoded"
                        .to_string(),
                ))
            }
        };

        Ok(Self {
            width,
            height,
            x_shift,
            y_shift,
        })
    }

    /// Geometry with explicit chroma shifts, for a layout that is not a Y4M
    /// chroma mode (NV12, packed 4:2:2, a decoded MJPEG frame, ...).
    ///
    /// Infallible, unlike [`PlaneGeometry::new`]: there is no alpha-plane
    /// case to reject, because the caller has already decided what the shifts
    /// are.
    pub(super) fn with_shifts(width: usize, height: usize, x_shift: u32, y_shift: u32) -> Self {
        Self {
            width,
            height,
            x_shift,
            y_shift,
        }
    }

    /// Geometry of a packed or single-plane picture: no chroma subsampling.
    ///
    /// Only the dimensions matter for such a layout, and they are all
    /// [`resize_packed`] reads.
    pub(super) fn packed(width: usize, height: usize) -> Self {
        Self::with_shifts(width, height, 0, 0)
    }

    /// Chroma plane width: `ceil(width / 2^x_shift)`, the Y4M rule.
    pub(super) fn chroma_width(&self) -> usize {
        ceil_shr(self.width, self.x_shift)
    }

    /// Chroma plane height: `ceil(height / 2^y_shift)`, the Y4M rule.
    pub(super) fn chroma_height(&self) -> usize {
        ceil_shr(self.height, self.y_shift)
    }

    /// Luma plane length in bytes.
    fn luma_len(&self) -> usize {
        self.width.saturating_mul(self.height)
    }

    /// Length in bytes of one chroma plane (there are two).
    ///
    /// Meaningless for `Mono`, which has no chroma planes at all: with both
    /// shifts at 0 this reports a full luma-sized plane. `convert_mono`
    /// never calls it, and any future caller must branch on the chroma mode
    /// before trusting it.
    fn chroma_len(&self) -> usize {
        self.chroma_width().saturating_mul(self.chroma_height())
    }
}

/// Requested output picture: format and post-resize dimensions.
#[derive(Clone, Copy)]
pub(super) struct FrameTarget {
    /// Output pixel format.
    pub(super) format: PixelFormat,
    /// Output width in pixels, after any rescale.
    pub(super) width: usize,
    /// Output height in pixels, after any rescale.
    pub(super) height: usize,
}

impl FrameTarget {
    /// Describe an output picture.
    pub(super) fn new(format: PixelFormat, width: usize, height: usize) -> Self {
        Self {
            format,
            width,
            height,
        }
    }
}

/// `ceil(value / 2^shift)`.
fn ceil_shr(value: usize, shift: u32) -> usize {
    value.saturating_add((1usize << shift) - 1) >> shift
}

/// Convert one raw Y4M picture into packed HWC bytes in `target.format`,
/// rescaled to `target`'s dimensions.
fn convert_y4m_frame(
    raw: &[u8],
    chroma: Y4mChroma,
    geometry: PlaneGeometry,
    target: FrameTarget,
) -> IoResult<Vec<u8>> {
    match chroma {
        Y4mChroma::Mono => convert_mono(raw, geometry, target),
        Y4mChroma::C420jpeg
        | Y4mChroma::C420mpeg2
        | Y4mChroma::C420paldv
        | Y4mChroma::C422
        | Y4mChroma::C444 => convert_planar(raw, geometry, target),
        // Unreachable in practice: `PlaneGeometry::new` rejects this at open
        // time, so no `PureReader` holding a C444alpha source can exist. Kept
        // exhaustive (rather than `_ =>`) so a new chroma mode upstream is a
        // compile error here instead of a silent misdecode.
        Y4mChroma::C444alpha => Err(IoError::Unsupported(
            "Y4M C444alpha is not supported by the pure-Rust video backend".to_string(),
        )),
    }
}

/// Monochrome (`Cmono`) Y4M: a bare luma plane, no chroma.
fn convert_mono(raw: &[u8], geometry: PlaneGeometry, target: FrameTarget) -> IoResult<Vec<u8>> {
    let luma = plane(raw, 0, geometry.luma_len(), "Y")?;

    // NOTE: identity copy, deliberately -- no BT.601 limited-to-full range
    // expansion. A Y4M luma sample is nominally studio range (16..=235), and
    // expanding it here would shift every sample by several code values. The
    // FFmpeg backend does not expand either (libswscale maps GRAY8 -> GRAY8
    // straight through), and the parity test asserts a decoded sample is
    // within +/-2 of the byte written into the file -- range expansion would
    // miss that by an order of magnitude. `yuv420p_to_gray8` is OxiMedia's
    // name for exactly this: hand back the luma plane untouched.
    let gray = yuv420p_to_gray8(luma, geometry.width, geometry.height);

    match target.format {
        PixelFormat::Gray => resize_packed(gray, geometry, target, 1),
        PixelFormat::Rgb | PixelFormat::Rgba => {
            // Resize FIRST here: the source is a single 8-bit plane, so
            // rescaling before the 1 -> 3 channel expansion touches a third
            // of the bytes. Safe in this case precisely because there is no
            // chroma plane that could fall out of step with luma (contrast
            // `convert_planar`, which must convert first).
            let scaled = resize_packed(gray, geometry, target, 1)?;
            let rgb = gray8_to_rgb24(&scaled, target.width, target.height);
            if matches!(target.format, PixelFormat::Rgba) {
                Ok(rgb24_to_rgba32(&rgb))
            } else {
                Ok(rgb)
            }
        }
    }
}

/// Planar Y'CbCr Y4M (`C420*`, `C422`, `C444`): luma plane followed by two
/// chroma planes.
fn convert_planar(raw: &[u8], geometry: PlaneGeometry, target: FrameTarget) -> IoResult<Vec<u8>> {
    let luma_len = geometry.luma_len();
    let chroma_len = geometry.chroma_len();
    let luma = plane(raw, 0, luma_len, "Y")?;

    match target.format {
        PixelFormat::Gray => {
            // Chroma is dropped: the luma plane *is* the greyscale image.
            // Same identity, no-range-expansion rule as `convert_mono`, and
            // the same thing libswscale does for yuv420p -> GRAY8.
            let gray = yuv420p_to_gray8(luma, geometry.width, geometry.height);
            resize_packed(gray, geometry, target, 1)
        }
        PixelFormat::Rgb | PixelFormat::Rgba => {
            let u = plane(raw, luma_len, chroma_len, "Cb")?;
            let v = plane(raw, luma_len.saturating_add(chroma_len), chroma_len, "Cr")?;

            // Convert BEFORE rescaling for a subsampled source. The chroma
            // planes are sampled at a different rate from luma, so resizing
            // each plane independently resamples them on different grids and
            // the two drift apart -- colour smears across edges. Upsampling
            // chroma to luma resolution first (which is what the YUV -> RGB
            // step does) leaves one packed image that resamples cleanly.
            let rgb = planar_to_rgb24(luma, u, v, geometry)?;
            let scaled = resize_packed(rgb, geometry, target, 3)?;
            if matches!(target.format, PixelFormat::Rgba) {
                Ok(rgb24_to_rgba32(&scaled))
            } else {
                Ok(scaled)
            }
        }
    }
}

/// Planar Y'CbCr to packed RGB24 with ITU-R BT.601 coefficients, for any
/// chroma subsampling `geometry` describes.
///
/// Chroma is upsampled by replication (nearest sample), which is what
/// `oximedia-core`'s own 4:2:0 converter does and what libswscale's default
/// fast path does.
pub(super) fn planar_to_rgb24(
    luma: &[u8],
    u: &[u8],
    v: &[u8],
    geometry: PlaneGeometry,
) -> IoResult<Vec<u8>> {
    let width = geometry.width;
    let height = geometry.height;

    if width == 0 || height == 0 {
        return Err(IoError::ConfigError(
            "Video frame dimensions must be non-zero".into(),
        ));
    }

    // Fast path onto OxiMedia's own 4:2:0 converter. Its plane-size asserts
    // use `(width / 2) * (height / 2)` (truncating) while Y4M sizes chroma
    // planes with `((width + 1) / 2) * ((height + 1) / 2)` (rounding up), so
    // the two agree only for even dimensions -- for odd ones that assert
    // would fire (a panic) and its `u[(y / 2) * (width / 2) + x / 2]`
    // indexing would run off the end of the plane. `planar_to_rgb24_generic`
    // covers those, and every non-4:2:0 mode, producing bit-identical output
    // (asserted by `upstream_and_generic_420_paths_agree` below): it drives
    // the same `PixelConverter::new(ColorMatrix::Bt601)` with the same
    // replicated-chroma lookup.
    if geometry.x_shift == 1
        && geometry.y_shift == 1
        && width.is_multiple_of(2)
        && height.is_multiple_of(2)
        && luma.len() == width.saturating_mul(height)
        && u.len() == (width / 2).saturating_mul(height / 2)
        && v.len() == u.len()
    {
        return Ok(yuv420p_to_rgb24(
            luma,
            u,
            v,
            width,
            height,
            ColorMatrix::Bt601,
        ));
    }

    planar_to_rgb24_generic(luma, u, v, geometry)
}

/// Planar Y'CbCr to packed RGB24 for any subsampling, without the
/// even-dimension precondition `oximedia-core`'s 4:2:0 converter carries.
///
/// Split out from [`planar_to_rgb24`] so the two can be compared directly in
/// tests; see that function for why both exist.
pub(super) fn planar_to_rgb24_generic(
    luma: &[u8],
    u: &[u8],
    v: &[u8],
    geometry: PlaneGeometry,
) -> IoResult<Vec<u8>> {
    let width = geometry.width;
    let height = geometry.height;
    let chroma_width = geometry.chroma_width();
    let chroma_height = geometry.chroma_height();

    if width == 0 || height == 0 {
        return Err(IoError::ConfigError(
            "Video frame dimensions must be non-zero".into(),
        ));
    }

    let converter = PixelConverter::new(ColorMatrix::Bt601);
    let row_bytes = width.checked_mul(3).ok_or_else(|| {
        IoError::ConfigError(format!("Y4M row of {width} pixels overflows usize"))
    })?;
    let mut rgb = vec![0u8; row_bytes.saturating_mul(height)];

    for (row, out_row) in rgb.chunks_exact_mut(row_bytes).enumerate() {
        let luma_row = row_of(luma, row, width, "Y")?;
        let chroma_row = (row >> geometry.y_shift).min(chroma_height.saturating_sub(1));
        let u_row = row_of(u, chroma_row, chroma_width, "Cb")?;
        let v_row = row_of(v, chroma_row, chroma_width, "Cr")?;

        // `out_row` is exactly `width` 3-byte pixels and `luma_row` exactly
        // `width` samples, so `col` indexes both in range; `chroma_col` is
        // clamped below `chroma_width`, the length of `u_row`/`v_row`. Every
        // index below is therefore in bounds by construction.
        for (col, out) in out_row.chunks_exact_mut(3).enumerate() {
            let chroma_col = (col >> geometry.x_shift).min(chroma_width.saturating_sub(1));
            let (r, g, b) =
                converter.yuv_to_rgb(luma_row[col], u_row[chroma_col], v_row[chroma_col]);
            out[0] = r;
            out[1] = g;
            out[2] = b;
        }
    }

    Ok(rgb)
}

/// Borrow `len` bytes of a plane starting at `offset`, or report the frame
/// as truncated.
///
/// `Y4mDemuxer` sizes every frame from the header, so a short plane means
/// the header and the payload disagree -- an error, never a panic.
fn plane<'a>(raw: &'a [u8], offset: usize, len: usize, label: &str) -> IoResult<&'a [u8]> {
    let end = offset.checked_add(len).ok_or_else(|| {
        IoError::ReadFailed(format!("Y4M {label} plane length {len} overflows usize"))
    })?;
    raw.get(offset..end).ok_or_else(|| {
        IoError::ReadFailed(format!(
            "Truncated Y4M frame: the {label} plane needs bytes {offset}..{end} of a \
             {}-byte frame",
            raw.len()
        ))
    })
}

/// Borrow row `row` (of `width` samples) from a single-channel plane.
fn row_of<'a>(data: &'a [u8], row: usize, width: usize, label: &str) -> IoResult<&'a [u8]> {
    let start = row.checked_mul(width).ok_or_else(|| {
        IoError::ReadFailed(format!("Y4M {label} plane row offset overflows usize"))
    })?;
    plane(data, start, width, label)
}

/// Rescale packed `channels`-per-pixel data to the target dimensions with
/// bilinear interpolation -- the same filter the FFmpeg backend asks
/// libswscale for (`Flags::BILINEAR`).
///
/// Returns the input untouched when no rescale was requested, so the common
/// "decode at native resolution" path costs nothing and stays bit-exact.
pub(super) fn resize_packed(
    data: Vec<u8>,
    geometry: PlaneGeometry,
    target: FrameTarget,
    channels: usize,
) -> IoResult<Vec<u8>> {
    if geometry.width == target.width && geometry.height == target.height {
        return Ok(data);
    }

    let config = ResizeConfig::new(
        dimension_as_u32(target.width)?,
        dimension_as_u32(target.height)?,
        ResizeMethod::Bilinear,
        channels,
    );
    resize_image(
        &data,
        dimension_as_u32(geometry.width)?,
        dimension_as_u32(geometry.height)?,
        &config,
    )
    .map_err(|e| {
        IoError::ReadFailed(format!(
            "Rescale to {}x{} failed: {e}",
            target.width, target.height
        ))
    })
}

/// Narrow a pixel dimension to the `u32` the OxiMedia resampler takes.
fn dimension_as_u32(value: usize) -> IoResult<u32> {
    u32::try_from(value)
        .map_err(|_| IoError::ConfigError(format!("Frame dimension {value} does not fit in u32")))
}

/// Widen packed RGB24 to RGBA32 with a fully opaque alpha channel.
///
/// Y4M carries no alpha kizzasi can use (`C444alpha`, the one chroma mode
/// that has an alpha plane, is rejected at open time), so 255 is the only
/// honest value -- the same constant libswscale writes for RGB24 -> RGBA.
/// Candidate for upstreaming into `oximedia-core`'s `convert::pixel`, which
/// has `gray8_to_rgb24` and friends but no RGB -> RGBA widening yet.
pub(super) fn rgb24_to_rgba32(rgb: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
    for pixel in rgb.chunks_exact(3) {
        rgba.extend_from_slice(pixel);
        rgba.push(u8::MAX);
    }
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4:2:0 chroma planes are `ceil(w/2) x ceil(h/2)`, so an odd-sized
    /// picture must not reach `yuv420p_to_rgb24` -- whose `assert_eq!` on
    /// `(w/2)*(h/2)` would abort the process. The local loop must handle it.
    #[test]
    fn odd_sized_420_converts_without_panicking() {
        let geometry = PlaneGeometry::new(3, 3, Y4mChroma::C420jpeg).expect("420 geometry");
        assert_eq!(geometry.chroma_width(), 2);
        assert_eq!(geometry.chroma_height(), 2);

        let mut raw = vec![128u8; geometry.luma_len()];
        raw.extend(std::iter::repeat_n(128u8, geometry.chroma_len() * 2));

        let target = FrameTarget {
            format: PixelFormat::Rgb,
            width: 3,
            height: 3,
        };
        let rgb = convert_y4m_frame(&raw, Y4mChroma::C420jpeg, geometry, target)
            .expect("odd 4:2:0 must convert");
        assert_eq!(rgb.len(), 3 * 3 * 3);
    }

    /// The upstream 4:2:0 fast path and the generic loop must agree bit for
    /// bit, otherwise even- and odd-sized pictures would decode differently.
    #[test]
    fn upstream_and_generic_420_paths_agree() {
        let geometry = PlaneGeometry::new(4, 4, Y4mChroma::C420jpeg).expect("420 geometry");
        let luma: Vec<u8> = (0..16u32).map(|i| (i * 13 + 20) as u8).collect();
        let u: Vec<u8> = (0..4u32).map(|i| (i * 31 + 90) as u8).collect();
        let v: Vec<u8> = (0..4u32).map(|i| (i * 17 + 140) as u8).collect();

        // `planar_to_rgb24` takes the upstream fast path for these even
        // dimensions; call the generic loop directly for the comparison.
        let upstream = planar_to_rgb24(&luma, &u, &v, geometry).expect("fast path");
        let generic = planar_to_rgb24_generic(&luma, &u, &v, geometry).expect("generic path");

        assert_eq!(upstream, generic);
        assert_eq!(upstream.len(), 4 * 4 * 3);
    }

    /// A frame shorter than its header promises must error, never panic.
    #[test]
    fn truncated_frame_is_an_error_not_a_panic() {
        let geometry = PlaneGeometry::new(4, 4, Y4mChroma::C420jpeg).expect("420 geometry");
        let target = FrameTarget {
            format: PixelFormat::Rgb,
            width: 4,
            height: 4,
        };
        let raw = vec![0u8; 8]; // needs 16 + 4 + 4
        assert!(convert_y4m_frame(&raw, Y4mChroma::C420jpeg, geometry, target).is_err());
    }

    /// `C444alpha` is refused up front rather than silently losing alpha.
    #[test]
    fn c444alpha_is_rejected_at_geometry_time() {
        assert!(matches!(
            PlaneGeometry::new(4, 4, Y4mChroma::C444alpha),
            Err(IoError::Unsupported(_))
        ));
    }

    #[test]
    fn rgba_widening_is_opaque() {
        let rgba = rgb24_to_rgba32(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(rgba, vec![1, 2, 3, 255, 4, 5, 6, 255]);
    }

    #[test]
    fn chroma_geometry_matches_the_y4m_rule() {
        for (chroma, expected) in [
            (Y4mChroma::C420jpeg, (4usize, 4usize)),
            (Y4mChroma::C422, (4, 7)),
            (Y4mChroma::C444, (7, 7)),
        ] {
            let geometry = PlaneGeometry::new(7, 7, chroma).expect("geometry");
            assert_eq!(
                (geometry.chroma_width(), geometry.chroma_height()),
                expected,
                "chroma {chroma}"
            );
            // Must agree with the demuxer's own frame sizing.
            assert_eq!(
                geometry.luma_len() + geometry.chroma_len() * 2,
                chroma.bytes_per_frame(7, 7),
                "frame size for chroma {chroma}"
            );
        }
    }
}
