//! Video frame input and processing
//!
//! Provides video stream input through a pluggable backend: FFmpeg (feature
//! `video`) for reading video files, network streams and camera devices,
//! and a pure-Rust backend (feature `video-pure`, the OxiMedia stack) that
//! reads Y4M files and captures from cameras without linking any C -- see
//! "Backend selection" below.
//!
//! ## Features
//!
//! - Read video files (with `video`: any container/codec the linked FFmpeg
//!   build supports -- MP4, AVI, MKV, WebM, Y4M, ...; with `video-pure`:
//!   Y4M) and network streams (RTSP/HTTP/..., FFmpeg only)
//! - Camera input: via `libavdevice` (v4l2, DirectShow, AVFoundation) under
//!   `video`, or via `oximedia-capture` (V4L2, AVFoundation, Media
//!   Foundation) under `video-pure`
//! - Frame decimation (skip frames for reduced processing)
//! - Frame buffering for smooth playback
//! - RGB / RGBA / Grayscale conversion and rescaling (libswscale, or
//!   `oximedia-core` + `oximedia-simd` + `oximedia-cv` under the pure
//!   backend)
//! - Seeking (`start_time`, [`VideoReader::seek`]) and `max_frames` limits.
//!   Neither applies to a live camera under the pure backend, which reports
//!   an honest [`IoError::Unsupported`](crate::error::IoError::Unsupported)
//!   rather than silently discarding frames off the front of the stream.
//!
//! Camera *enumeration* ([`CameraDevice::list_devices`]) depends on which
//! features are compiled in. With `video-pure` it goes through
//! `oximedia-capture` on **every** platform and reports real device names
//! and the modes each device advertises -- and that implementation is
//! preferred even in a build that also has `video`, because the FFmpeg one
//! cannot do the same. In a `video`-only build it is Linux-only
//! (`/dev/video*`) and returns
//! [`IoError::Unsupported`](crate::error::IoError::Unsupported) elsewhere
//! rather than inventing a device list. Opening a named camera device works
//! on every platform whose capture backend the chosen implementation
//! provides.
//!
//! ## Example
//!
//! ```rust,no_run
//! use kizzasi_io::{VideoReader, VideoConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Any container the `video` feature's FFmpeg build supports. With
//!     // only `video-pure` compiled in, point this at a `.y4m` path instead
//!     // -- see "Backend selection" below for the exact per-source routing.
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
//!
//! ## Backend selection
//!
//! [`VideoReader::new`] does not talk to FFmpeg (or anything else) directly;
//! it opens the configured source through whichever backend
//! [`VideoConfig::backend`] ([`VideoBackend`]) resolves to:
//!
//! - [`VideoBackend::Ffmpeg`] and [`VideoBackend::Pure`] are explicit
//!   requests. Each succeeds only when its feature (`video` / `video-pure`
//!   respectively) is compiled in; forcing a backend never silently falls
//!   back to the other one.
//! - [`VideoBackend::Auto`] (the default) picks per source: prefer the
//!   pure-Rust backend where it reports itself capable, otherwise fall back
//!   to FFmpeg, and error only if neither is usable.
//!
//! With both features compiled in, that resolves per source exactly as
//! follows (with only one of the two features compiled in, only that
//! feature's row applies, and `Auto` errors -- naming what is missing --
//! if the source needs the other one):
//!
//! | Source | `Auto` resolves to |
//! |---|---|
//! | Y4M (`YUV4MPEG2`) file | Pure, whenever `video-pure` is compiled in; FFmpeg (`video`) otherwise |
//! | Any other file (MP4, AVI, MKV, WebM, ...) | FFmpeg (`video`) -- the pure backend decodes Y4M only |
//! | Camera | Pure, on a platform `oximedia-capture` has a backend for (Linux, macOS, Windows); FFmpeg (`video`) otherwise |
//! | Network stream (RTSP/HTTP/...) | FFmpeg (`video`) only -- the pure backend has no network demuxer |
//!
//! **What the pure backend covers today:**
//!
//! * **Y4M (YUV4MPEG2) files**, decoded end to end by the OxiMedia stack --
//!   `oximedia-container` demuxes, `oximedia-core` converts the planes to
//!   the configured [`PixelFormat`] and `oximedia-cv` rescales.
//! * **Camera capture**, on any platform `oximedia-capture` has a backend
//!   for -- V4L2 on Linux, AVFoundation on macOS, Media Foundation on
//!   Windows -- `oximedia-capture` negotiates a mode and delivers frames,
//!   `oximedia-codec` decodes MJPEG ones, `oximedia-simd` converts packed
//!   4:2:2 rows, and the same `oximedia-core`/`oximedia-cv` pair finishes
//!   the job.
//!
//! Its capability probe (`backend_pure::PureReader::supports`) answers
//! `true` for a file carrying the `YUV4MPEG2` magic and for a camera on a
//! platform with a capture backend, and `false` for everything else -- which
//! is exactly the table above. Forcing [`VideoBackend::Pure`] on a source
//! the probe answers `false` for returns an honest [`IoError::Unsupported`]
//! naming what the pure backend does support, rather than falling back to
//! FFmpeg.
//!
//! **What the pure backend does *not* do:**
//!
//! * **Patent-encumbered codecs.** H.264/AVC, H.265/HEVC, H.266/VVC and AAC
//!   are on OxiMedia's own "Red List" -- excluded permanently on
//!   patent-encumbrance grounds, not merely unimplemented yet. A file using
//!   any of them needs the `video` (FFmpeg) feature; no future `video-pure`
//!   version will decode them.
//! * **Containers other than Y4M.** `oximedia-container` has demuxers for
//!   several (MP4, MKV, ...), but no decoders are wired into this backend,
//!   so claiming support would mean claiming to decode whatever codec is
//!   inside. Y4M is the only file format this backend opens today.
//! * **Network streams.** No pure-Rust RTSP/HTTP demuxer is wired up; see
//!   the routing table above.
//! * **Seeking a live camera.** [`VideoReader::seek`] and `start_time` are
//!   refused with an honest [`IoError::Unsupported`] rather than silently
//!   discarding frames off the front of the stream, and `frame_count`/
//!   `duration` are reported as `0`. A live camera is not a file, and the
//!   pure backend does not pretend otherwise.
//!
//! **Verification status.** Y4M decoding and camera capture both run
//! against deterministic, scripted fixtures on every host this crate is
//! tested on: the reader-parity suite (below) runs against synthetically
//! generated Y4M files, and the camera suite drives
//! `oximedia_capture::mock`, a scripted capture backend that never touches
//! real hardware or an OS permission prompt. Real-hardware coverage is a
//! narrower claim. While preparing this documentation,
//! `CameraDevice::list_devices()` was run against the real platform API on
//! this workspace's macOS development machine and correctly reached
//! AVFoundation and surfaced a genuine TCC permission error
//! ([`IoError::Connection`](crate::error::IoError::Connection), message
//! naming TCC) rather than fabricating a device list or panicking -- that
//! is evidence the integration is wired to the real capture API, *not*
//! evidence that a real camera was enumerated (this machine has not granted
//! the process Camera access). Linux (V4L2) and Windows (Media Foundation)
//! have not been exercised against real hardware from this repository.
//! Real-device smoke testing -- actually opening a camera and reading
//! frames from it -- is left to whoever runs this on hardware with a
//! camera attached and permission granted; see `TODO.md`.
//!
//! The two backends are held to the same observable contract for a source
//! both can open -- same frame indices, timestamps, decimation, buffering
//! and `max_frames` accounting -- and `reader_tests` runs its reader suite
//! once per compiled-in backend with identical assertions to enforce it.
//! `Auto`'s routing itself, not just the capability probe that feeds it, is
//! checked observably (both a forced-`Pure` and an `Auto` read of the same
//! Y4M file must decode to identical frames) by
//! `reader_tests::test_auto_routes_y4m_to_pure_observably`.
//!
//! ## Module layout
//!
//! Internally this module is split across a few private files (the flat
//! `kizzasi_io::Video*` paths above are the only supported public surface):
//!
//! - `types` -- value types: [`VideoFrame`], [`PixelFormat`], [`VideoSource`],
//!   [`VideoConfig`], [`VideoBackend`], [`CameraDevice`], [`VideoMetadata`],
//!   [`FrameBuffer`].
//! - `processing` -- [`OpticalFlow`]/[`OpticalFlowEstimator`] motion
//!   estimation and the [`VideoProcessor`]/[`VideoFilter`] spatial filters.
//!   Pure computation: no FFmpeg dependency, compiled under either backend
//!   feature.
//! - `backend_ffmpeg` (feature `video`) -- `FfmpegReader`, the only place
//!   that touches `ffmpeg_next`. A previous version of this reader never
//!   opened anything: `new()` returned `Ok` after only calling
//!   `ffmpeg::init()`, `read_frame()` always returned `Ok(None)` (immediate
//!   end-of-stream against a perfectly valid file) and `metadata()`
//!   reported a hardcoded 30 fps, 0 s duration and 1920x1080 for every
//!   source. See `FfmpegReader`'s own docs for what it actually does today.
//! - `backend_pure` (feature `video-pure`) -- `PureReader`, the pure-Rust
//!   (OxiMedia) backend described above: the Y4M reader, the reader shell
//!   every source shares, and the conversion/rescale helpers.
//! - `backend_pure_camera` (feature `video-pure`) -- the pure backend's
//!   live-capture half: opening a device, the `camera_format` alias table,
//!   `CaptureError` mapping and the capture pixel-layout conversions. Split
//!   from `backend_pure` for size, not layering; between them the two are
//!   the only place that touches `oximedia_container` / `oximedia_core` /
//!   `oximedia_cv` / `oximedia_capture` / `oximedia_codec` /
//!   `oximedia_simd`.
//! - This file (`mod.rs`) -- the [`VideoReader`] facade that dispatches to
//!   whichever backend `resolve_backend` picks, plus the backend selection
//!   rules themselves.

#[cfg(feature = "video")]
mod backend_ffmpeg;
#[cfg(feature = "video-pure")]
mod backend_pure;
#[cfg(feature = "video-pure")]
mod backend_pure_camera;
mod processing;
mod types;

#[cfg(test)]
mod reader_tests;

pub use processing::{
    OpticalFlow, OpticalFlowEstimator, OpticalFlowMethod, VideoFilter, VideoProcessor,
};
pub use types::{
    CameraDevice, FrameBuffer, PixelFormat, VideoBackend, VideoConfig, VideoFrame, VideoMetadata,
    VideoSource,
};

use crate::error::{IoError, IoResult};
use std::path::Path;

/// Video reader for streaming video frames.
///
/// Dispatches to a concrete backend chosen by [`VideoConfig::backend`] --
/// see the module-level "Backend selection" docs for the exact rules.
/// Every method below has the same signature and the same contract
/// regardless of which backend ends up handling the source.
pub struct VideoReader {
    inner: ReaderImpl,
}

/// The concrete backend behind a [`VideoReader`].
///
/// Each variant is compiled in only when its feature is enabled, so a
/// `video-pure`-only build never links FFmpeg and a `video`-only build
/// never pulls in the OxiMedia crates.
///
/// Both variants are boxed: `FfmpegReader` carries the whole FFmpeg
/// demuxer/decoder/scaler pipeline and `PureReader` a `VideoConfig`, a
/// buffered file handle and a cached Y4M header -- a few hundred bytes
/// each, and not the same few hundred. Boxing keeps `ReaderImpl` (and so
/// `VideoReader`) pointer-sized instead of padding every value out to the
/// larger variant, which is what `clippy::large_enum_variant` asks for.
enum ReaderImpl {
    #[cfg(feature = "video")]
    Ffmpeg(Box<backend_ffmpeg::FfmpegReader>),
    #[cfg(feature = "video-pure")]
    Pure(Box<backend_pure::PureReader>),
}

impl VideoReader {
    /// Open `config.source` with the backend [`VideoConfig::backend`]
    /// resolves to (see the module-level "Backend selection" docs).
    ///
    /// # Errors
    ///
    /// * [`IoError::Unsupported`] if the requested backend (or, for `Auto`,
    ///   every backend capable of the source) is not compiled into this
    ///   build.
    /// * [`IoError::ReadFailed`] if the source is a file that does not
    ///   exist.
    /// * Whatever error the chosen backend's own `new` reports otherwise.
    pub async fn new(config: VideoConfig) -> IoResult<Self> {
        let inner = match resolve_backend(&config)? {
            ResolvedBackend::Ffmpeg => Self::open_ffmpeg(config).await?,
            ResolvedBackend::Pure => Self::open_pure(config).await?,
        };
        Ok(Self { inner })
    }

    #[cfg(feature = "video")]
    async fn open_ffmpeg(config: VideoConfig) -> IoResult<ReaderImpl> {
        Ok(ReaderImpl::Ffmpeg(Box::new(
            backend_ffmpeg::FfmpegReader::new(config).await?,
        )))
    }

    // `resolve_backend` only ever returns `ResolvedBackend::Ffmpeg` when
    // `cfg!(feature = "video")` is true, so this arm is unreachable in
    // practice; it exists so `VideoReader::new` type-checks in a
    // `video-pure`-only build. Reported as a real error rather than
    // `unreachable!()` so a future bug in `resolve_backend` fails loudly
    // instead of panicking.
    #[cfg(not(feature = "video"))]
    async fn open_ffmpeg(_config: VideoConfig) -> IoResult<ReaderImpl> {
        Err(IoError::Unsupported(
            "resolved to the FFmpeg backend, but this build does not have the `video` \
             feature enabled"
                .to_string(),
        ))
    }

    #[cfg(feature = "video-pure")]
    async fn open_pure(config: VideoConfig) -> IoResult<ReaderImpl> {
        Ok(ReaderImpl::Pure(Box::new(
            backend_pure::PureReader::new(config).await?,
        )))
    }

    // See `open_ffmpeg`'s `not(feature = "video")` twin above: unreachable
    // in practice, handled as an honest error rather than a panic.
    #[cfg(not(feature = "video-pure"))]
    async fn open_pure(_config: VideoConfig) -> IoResult<ReaderImpl> {
        Err(IoError::Unsupported(
            "resolved to the pure-Rust backend, but this build does not have the \
             `video-pure` feature enabled"
                .to_string(),
        ))
    }

    /// Read the next frame.
    ///
    /// Returns `Ok(None)` only at genuine end-of-stream (or once the
    /// configured `max_frames` has been delivered); see the active
    /// backend's own docs for anything more specific.
    pub async fn read_frame(&mut self) -> IoResult<Option<VideoFrame>> {
        match &mut self.inner {
            #[cfg(feature = "video")]
            ReaderImpl::Ffmpeg(reader) => reader.read_frame().await,
            #[cfg(feature = "video-pure")]
            ReaderImpl::Pure(reader) => reader.read_frame().await,
        }
    }

    /// Get video metadata (fps, duration, dimensions, pixel format).
    pub fn metadata(&self) -> VideoMetadata {
        match &self.inner {
            #[cfg(feature = "video")]
            ReaderImpl::Ffmpeg(reader) => reader.metadata(),
            #[cfg(feature = "video-pure")]
            ReaderImpl::Pure(reader) => reader.metadata(),
        }
    }

    /// Seek to a specific time (seconds) in the source.
    pub async fn seek(&mut self, time: f64) -> IoResult<()> {
        match &mut self.inner {
            #[cfg(feature = "video")]
            ReaderImpl::Ffmpeg(reader) => reader.seek(time).await,
            #[cfg(feature = "video-pure")]
            ReaderImpl::Pure(reader) => reader.seek(time).await,
        }
    }

    /// Number of frames delivered by [`VideoReader::read_frame`] so far.
    pub fn current_frame(&self) -> u64 {
        match &self.inner {
            #[cfg(feature = "video")]
            ReaderImpl::Ffmpeg(reader) => reader.current_frame(),
            #[cfg(feature = "video-pure")]
            ReaderImpl::Pure(reader) => reader.current_frame(),
        }
    }

    /// Presentation timestamp (seconds) of the most recently delivered
    /// frame.
    pub fn current_time(&self) -> f64 {
        match &self.inner {
            #[cfg(feature = "video")]
            ReaderImpl::Ffmpeg(reader) => reader.current_time(),
            #[cfg(feature = "video-pure")]
            ReaderImpl::Pure(reader) => reader.current_time(),
        }
    }
}

/// The concrete backend [`resolve_backend`] picked for a [`VideoConfig`].
enum ResolvedBackend {
    /// Use the FFmpeg backend (`backend_ffmpeg`, feature `video`).
    Ffmpeg,
    /// Use the pure-Rust backend (`backend_pure`, feature `video-pure`).
    Pure,
}

/// Decide which backend a [`VideoReader::new`] call should use, without
/// constructing anything.
///
/// See the module-level "Backend selection" docs for the rules this
/// implements: [`VideoBackend::Ffmpeg`] / [`VideoBackend::Pure`] are
/// explicit requests that fail if their feature is not compiled in;
/// [`VideoBackend::Auto`] delegates to [`resolve_auto`].
///
/// # Errors
///
/// [`IoError::Unsupported`] if the requested backend is not compiled into
/// this build. See [`resolve_auto`] for the errors `Auto` can add on top.
fn resolve_backend(config: &VideoConfig) -> IoResult<ResolvedBackend> {
    match config.backend {
        VideoBackend::Ffmpeg => {
            if cfg!(feature = "video") {
                Ok(ResolvedBackend::Ffmpeg)
            } else {
                Err(IoError::Unsupported(
                    "VideoBackend::Ffmpeg was requested, but this build does not have \
                     the `video` feature enabled; enable the `video` feature to use the \
                     FFmpeg backend"
                        .to_string(),
                ))
            }
        }
        VideoBackend::Pure => {
            if cfg!(feature = "video-pure") {
                Ok(ResolvedBackend::Pure)
            } else {
                Err(IoError::Unsupported(
                    "VideoBackend::Pure was requested, but this build does not have the \
                     `video-pure` feature enabled; enable the `video-pure` feature to \
                     use the pure-Rust backend"
                        .to_string(),
                ))
            }
        }
        VideoBackend::Auto => resolve_auto(config),
    }
}

/// `Auto` backend resolution: prefer the pure backend where it reports
/// itself capable of the source, otherwise fall back to FFmpeg; error only
/// if neither is usable.
///
/// In practice that means: a Y4M file goes to the pure backend whenever
/// `video-pure` is compiled in, a camera goes there too on a platform
/// `oximedia-capture` has a backend for, and everything else -- other
/// containers, network streams, cameras on a platform without a pure capture
/// backend -- goes to FFmpeg, because those are precisely what
/// `backend_pure`'s probe answers `false` for. With only `video` enabled the
/// probe is a compile-time `false` and `Auto` behaves exactly as it did
/// before `video-pure` existed. New pure-backend sources arrive by changing
/// that probe (and implementing the source), not this function.
///
/// For a [`VideoSource::File`], the first 9 bytes are read up front to
/// check for the `YUV4MPEG2` magic that identifies a raw Y4M container --
/// unconditionally, regardless of which backend features are enabled, so a
/// missing file is reported the same way ([`IoError::ReadFailed`]) no
/// matter what is compiled in.
fn resolve_auto(config: &VideoConfig) -> IoResult<ResolvedBackend> {
    match &config.source {
        VideoSource::File(path) => {
            // `pure_backend_supports` re-runs this same probe internally.
            // `&&` short-circuits, so the second read happens only for a
            // file that really is Y4M (9 bytes off a regular file, once per
            // `VideoReader::new`), and never for the common non-Y4M case.
            if probe_is_y4m(path)? && pure_backend_supports(&config.source) {
                Ok(ResolvedBackend::Pure)
            } else if cfg!(feature = "video") {
                Ok(ResolvedBackend::Ffmpeg)
            } else {
                Err(IoError::Unsupported(format!(
                    "no usable video backend is compiled in to open {}; enable the \
                     `video` feature (the pure-Rust backend reads Y4M files only)",
                    path.display()
                )))
            }
        }
        VideoSource::Camera(device) => {
            if pure_backend_supports(&config.source) {
                Ok(ResolvedBackend::Pure)
            } else if cfg!(feature = "video") {
                Ok(ResolvedBackend::Ffmpeg)
            } else {
                Err(IoError::Unsupported(format!(
                    "no usable video backend is compiled in to open camera '{device}'; \
                     enable the `video` feature, or `video-pure` on a platform whose \
                     capture API oximedia-capture implements"
                )))
            }
        }
        VideoSource::Network(url) => {
            if cfg!(feature = "video") {
                Ok(ResolvedBackend::Ffmpeg)
            } else {
                Err(IoError::Unsupported(format!(
                    "no usable video backend is compiled in to open network stream \
                     '{url}'; the pure-Rust backend does not support network streams \
                     yet, enable the `video` feature"
                )))
            }
        }
    }
}

/// Whether the pure backend currently claims it can handle `source`.
///
/// Delegates to `backend_pure::PureReader::supports`, which today answers
/// `true` for a Y4M file and `false` for everything else -- see that module
/// for the full list of what is still missing.
#[cfg(feature = "video-pure")]
fn pure_backend_supports(source: &VideoSource) -> bool {
    backend_pure::PureReader::supports(source)
}

/// Twin of the `video-pure` version above: no pure backend is compiled in
/// at all, so it can never support anything.
#[cfg(not(feature = "video-pure"))]
fn pure_backend_supports(_source: &VideoSource) -> bool {
    false
}

/// Read the first 9 bytes of a file and report whether they are the
/// `YUV4MPEG2` magic that identifies a raw Y4M container.
///
/// A missing file is reported as [`IoError::ReadFailed`] with the same
/// "Video file not found" wording the FFmpeg backend has always used for a
/// bad path, rather than silently falling through to backend selection.
/// Any other reason the magic bytes cannot be read (permission, a
/// directory, a file shorter than 9 bytes, not a regular file at all, ...)
/// is reported as "not Y4M" rather than a hard error, so a backend still
/// gets the chance to make sense of the source the way it always has.
///
/// Only regular files are actually read. A FIFO / named pipe / character
/// device passed as `VideoSource::File` is non-seekable: opening it can
/// block until a writer appears, and the 9 bytes this probe consumes would
/// be gone by the time the chosen backend opens the same path -- for
/// `Auto`, the default, that would silently eat the start of the stream
/// without the caller having asked for probing at all. Skipping the read
/// for anything `Metadata::is_file` doesn't confirm keeps the probe
/// side-effect-free on such sources.
fn probe_is_y4m(path: &Path) -> IoResult<bool> {
    use std::io::Read;

    if !path.exists() {
        return Err(IoError::ReadFailed(format!(
            "Video file not found: {}",
            path.display()
        )));
    }

    if !matches!(std::fs::metadata(path), Ok(meta) if meta.is_file()) {
        return Ok(false);
    }

    let mut magic = [0u8; 9];
    let is_y4m = std::fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut magic))
        .is_ok()
        && &magic == b"YUV4MPEG2";

    Ok(is_y4m)
}
