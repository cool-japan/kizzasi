//! Camera capture for the pure-Rust video backend.
//!
//! Everything `backend_pure` needs in order to treat a live capture device
//! the way it already treats a Y4M file: `oximedia-capture` opens the device
//! and hands over frames, `oximedia-codec`'s `MjpegDecoder` decodes the
//! compressed ones, `oximedia-simd` converts packed 4:2:2 rows, and the
//! shared conversion helpers in `backend_pure` (`planar_to_rgb24`,
//! `resize_packed`, `rgb24_to_rgba32`, ...) finish the job -- the same code
//! that converts a Y4M picture, so a camera frame and a file frame reaching
//! the same [`PixelFormat`] go through the same arithmetic.
//!
//! Split out of `backend_pure.rs` purely for size: that file is the Y4M
//! reader plus the shared conversion library, this one is the capture path,
//! and together they would sit well over the 2000-line ceiling. Items shared
//! across the boundary are `pub(super)` in whichever file owns them.
//!
//! ## What "live" changes
//!
//! Three things behave differently from the file path, all of them because
//! the source is a device rather than bytes on disk:
//!
//! * **No seeking, and no `start_time`.** Both are rejected with
//!   [`IoError::Unsupported`]; there is no past to seek into, and silently
//!   dropping the first N seconds of a live stream is not a seek.
//! * **No batch pumping.** `backend_pure`'s Y4M pump fills the whole frame
//!   buffer before returning, which costs nothing for a file. Doing that to a
//!   camera would make the first `read_frame()` wait for `buffer_size` frame
//!   periods (a third of a second at the default 5 frames / 30 fps), so the
//!   camera pump takes exactly one frame per call. Frame indices,
//!   decimation and `max_frames` accounting are unchanged.
//! * **No frame count or duration.** `0` for both, the same honest "unknown"
//!   the Y4M path reports.
//!
//! ## Blocking receive inside `async fn`
//!
//! [`CameraSource::recv`] blocks. `oximedia-capture` runs its capture loop on
//! a dedicated OS thread and delivers through a bounded channel, so a blocked
//! consumer never deadlocks the producer -- and the FFmpeg backend blocks in
//! exactly the same place (`av_read_frame`), which is what the parity
//! contract is measured against. Wrapping this in `spawn_blocking` would mean
//! moving `CaptureStream` in and out of the reader on every frame for no
//! change in observable behaviour.

use super::backend_pure::{
    planar_to_rgb24, planar_to_rgb24_generic, resize_packed, rgb24_to_rgba32, FrameTarget,
    PlaneGeometry,
};
use super::types::{PixelFormat, VideoConfig};
use crate::error::{IoError, IoResult};
use oximedia_capture::{
    backend_kind, BackendKind, CaptureConfig, CaptureEncoding, CaptureError, CaptureFormat,
    CaptureFrame, CaptureSession, CaptureStream, DeviceSelector, DropPolicy, FramePayload,
};
use oximedia_codec::{MjpegDecoder, Plane, VideoDecoder, VideoFrame as CaptureVideoFrame};
use oximedia_core::convert::pixel::{gray8_to_rgb24, rgb24_to_gray8};
use oximedia_core::convert::simd_pixel::{nv12_to_rgb24, nv21_to_rgb24, SimdColorMatrix};
use oximedia_core::PixelFormat as CapturePixelFormat;
use oximedia_simd::yuv_ops::{convert_uyvy422_row_to_rgb, convert_yuv422_row_to_rgb};
use std::sync::Once;
use tracing::info;

// ============================================================================
// Camera input-format aliases
// ============================================================================

/// `camera_format` names the pure backend accepts, and the capture backend
/// each one selects.
///
/// The names are FFmpeg's `libavdevice` input-device names, because that is
/// what [`VideoConfig::camera_format`] has always carried and what
/// `CameraDevice::default_format()` still produces -- a config written for
/// the FFmpeg backend must mean the same thing here or it means nothing.
///
/// `dshow` and `vfwcap` are the interesting rows: DirectShow and Video for
/// Windows are both legacy Windows capture APIs that `oximedia-capture` does
/// not implement and will not, because Microsoft deprecated both in favour of
/// Media Foundation. Accepting them as *aliases for Media Foundation* (with a
/// one-time `info!`) is the honest reading of "capture on Windows"; refusing
/// them would break every existing Windows config for a difference the caller
/// cannot act on.
pub(super) const CAMERA_FORMAT_ALIASES: &[(&str, BackendKind)] = &[
    ("video4linux2", BackendKind::V4l2),
    ("v4l2", BackendKind::V4l2),
    ("avfoundation", BackendKind::AvFoundation),
    ("dshow", BackendKind::MediaFoundation),
    ("msmf", BackendKind::MediaFoundation),
    ("vfwcap", BackendKind::MediaFoundation),
];

/// Names that select the host's own backend rather than a specific one.
const HOST_ALIAS: &str = "auto";

/// Guards the one-time DirectShow-to-Media-Foundation notice.
static DIRECTSHOW_NOTICE: Once = Once::new();

/// Comma-separated list of every accepted alias, for error messages.
fn accepted_aliases() -> String {
    let mut names: Vec<&str> = CAMERA_FORMAT_ALIASES
        .iter()
        .map(|(name, _)| *name)
        .collect();
    names.push(HOST_ALIAS);
    names.join(", ")
}

/// Which capture backend a `camera_format` name asks for, checked against the
/// backend `host` actually provides.
///
/// `host` is a parameter rather than a call to [`backend_kind`] so that every
/// row of the table -- including the "asked for another platform's API" ones
/// -- is testable on any machine. [`resolve_camera_platform_for_host`] is the
/// production entry point.
///
/// # Errors
///
/// * [`IoError::Unsupported`] if `name` is not one of
///   [`CAMERA_FORMAT_ALIASES`] (or `auto`). The message lists what is
///   accepted.
/// * [`IoError::Unsupported`] if `host` has no capture backend compiled in
///   for this target -- `oximedia-capture` implements V4L2, AVFoundation and
///   Media Foundation, and reports [`BackendKind::Unsupported`] elsewhere
///   rather than pretending.
/// * [`IoError::Unsupported`] if the resolved backend is not the host's.
///   Running the AVFoundation capture path on Linux is not a thing that can
///   be arranged, and quietly opening a V4L2 device for a config that said
///   `avfoundation` would be a silent substitution -- the caller asked a
///   specific question and deserves a specific "no".
pub(super) fn resolve_camera_platform(
    name: Option<&str>,
    host: BackendKind,
) -> IoResult<BackendKind> {
    let requested = match name.map(str::trim).filter(|name| !name.is_empty()) {
        None => host,
        Some(name) if name.eq_ignore_ascii_case(HOST_ALIAS) => host,
        Some(name) => {
            let lowered = name.to_ascii_lowercase();
            let matched = CAMERA_FORMAT_ALIASES
                .iter()
                .find(|(alias, _)| *alias == lowered)
                .map(|(_, kind)| *kind);
            match matched {
                Some(kind) => {
                    if matches!(lowered.as_str(), "dshow" | "vfwcap") {
                        DIRECTSHOW_NOTICE.call_once(|| {
                            info!(
                                backend = "pure",
                                "camera_format '{lowered}' names a legacy Windows capture API; \
                                 the pure-Rust backend serves it through Media Foundation, \
                                 which is what oximedia-capture implements on Windows"
                            );
                        });
                    }
                    kind
                }
                None => {
                    return Err(IoError::Unsupported(format!(
                        "unknown camera input format '{name}' for the pure-Rust video \
                         backend; accepted names are: {}",
                        accepted_aliases()
                    )))
                }
            }
        }
    };

    if !host.is_available() {
        return Err(IoError::Unsupported(format!(
            "the pure-Rust video backend has no camera capture backend for {}: \
             oximedia-capture implements Video4Linux2 (Linux), AVFoundation (macOS) and \
             Media Foundation (Windows) -- enable the `video` (FFmpeg) feature to capture \
             on this platform",
            std::env::consts::OS
        )));
    }

    if requested != host {
        return Err(IoError::Unsupported(format!(
            "camera_format '{}' selects the {requested} capture backend, but this host \
             provides {host}; the pure-Rust video backend never substitutes one \
             platform's capture API for another's -- use '{HOST_ALIAS}' (or drop \
             camera_format) to take whatever this host offers",
            name.unwrap_or(HOST_ALIAS)
        )));
    }

    Ok(requested)
}

/// [`resolve_camera_platform`] against the backend this build actually has.
pub(super) fn resolve_camera_platform_for_host(name: Option<&str>) -> IoResult<BackendKind> {
    resolve_camera_platform(name, backend_kind())
}

/// Whether this build has a capture backend at all.
///
/// `backend_pure::PureReader::supports` consults this so that `Auto` keeps
/// routing cameras to FFmpeg on a target `oximedia-capture` has no backend
/// for, instead of claiming the source and then failing to open it.
pub(super) fn host_capture_is_available() -> bool {
    backend_kind().is_available()
}

// ============================================================================
// Error mapping
// ============================================================================

/// Map a [`CaptureError`] onto this crate's error type.
///
/// The mapping keeps the *kind* of failure legible, which is the only thing a
/// caller can act on:
///
/// | `CaptureError` | `IoError` | why |
/// |---|---|---|
/// | `UnsupportedPlatform` | `Unsupported` | no backend compiled in for this target |
/// | `DeviceNotFound` | `Connection` | the device is not there (same variant the FFmpeg backend reports for a camera that will not open) |
/// | `PermissionDenied` | `Connection` | the device is there but the process may not reach it; the message names the OS permission |
/// | `NoMatchingFormat` | `ConfigError` | the *request* (size / fps) is unsatisfiable -- relax it |
/// | `FormatRejected` | `ConfigError` | the device advertised a mode it will not deliver -- ask for another |
/// | `Io` | `ReadFailed` | an OS-level failure on an open device |
/// | `Platform` | `ReadFailed` | a backend fault with no portable shape |
/// | `ThreadPanic` | `ReadFailed` | the capture thread unwound; the session is finished |
/// | `Core` | `ReadFailed` | an `oximedia-core` failure surfaced through capture |
///
/// `CaptureError` is `#[non_exhaustive]`, so the catch-all arm is required;
/// it maps to `ReadFailed` and includes the upstream `Display` text verbatim
/// rather than inventing a summary for a variant this build has never seen.
pub(super) fn map_capture_error(error: CaptureError, device: &str) -> IoError {
    match error {
        CaptureError::UnsupportedPlatform { target } => IoError::Unsupported(format!(
            "the pure-Rust video backend cannot capture on {target}: oximedia-capture has \
             no backend for this platform -- enable the `video` (FFmpeg) feature"
        )),
        CaptureError::DeviceNotFound(selector) => IoError::Connection(format!(
            "camera '{device}' not found: no capture device matches {selector}"
        )),
        CaptureError::PermissionDenied { device: refused } => IoError::Connection(format!(
            "permission denied for camera '{refused}': {}",
            permission_hint()
        )),
        CaptureError::NoMatchingFormat {
            device: id,
            available,
        } => IoError::ConfigError(format!(
            "camera '{id}' advertises {available} format(s), none of which satisfy the \
             requested size/frame rate; relax target_width/target_height/camera_fps"
        )),
        CaptureError::FormatRejected {
            device: id,
            format,
            reason,
        } => IoError::ConfigError(format!(
            "camera '{id}' advertised {format} but refused to be configured for it: {reason}"
        )),
        CaptureError::Io { device: id, source } => {
            IoError::ReadFailed(format!("I/O error on camera '{id}': {source}"))
        }
        CaptureError::Platform(message) => {
            IoError::ReadFailed(format!("camera '{device}' backend error: {message}"))
        }
        CaptureError::ThreadPanic => IoError::ReadFailed(format!(
            "the capture thread for camera '{device}' panicked; the session is finished"
        )),
        CaptureError::Core(inner) => IoError::ReadFailed(format!(
            "camera '{device}' failed in oximedia-core: {inner}"
        )),
        other => IoError::ReadFailed(format!("camera '{device}' capture error: {other}")),
    }
}

/// Platform-specific wording for a refused capture device.
fn permission_hint() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "macOS gates camera access behind TCC -- grant this application (or the terminal \
         running it) Camera access in System Settings > Privacy & Security > Camera, then \
         restart the process"
    }
    #[cfg(target_os = "linux")]
    {
        "on Linux this is usually group membership -- the user must be in the `video` group \
         that owns the /dev/video* node"
    }
    #[cfg(target_os = "windows")]
    {
        "on Windows check Settings > Privacy & security > Camera and allow desktop apps to \
         access the camera"
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "the operating system refused access to the capture device"
    }
}

// ============================================================================
// The opened camera
// ============================================================================

/// A live capture device, opened and delivering frames.
///
/// Held by `backend_pure`'s `PureSource::Camera`. Dropping it drops the
/// [`CaptureSession`], which raises the stop flag and joins the capture
/// thread -- which is why `session` is kept even though every frame arrives
/// through `stream`.
pub(super) struct CameraSource {
    /// Owns the capture thread and the device handle. Dropping it stops both.
    session: CaptureSession,
    /// The delivery queue's consuming end.
    stream: CaptureStream,
    /// MJPEG decoder, built on the first compressed frame and reused after.
    ///
    /// Lazy because a device that negotiated a raw format never needs one, and
    /// `MjpegDecoder::new` allocates a JPEG decoder.
    mjpeg: Option<MjpegDecoder>,
    /// Frame rate of the negotiated mode, in frames per second.
    fps: f64,
    /// The mode the device was actually configured for -- the outcome of
    /// negotiation, not the request.
    negotiated: CaptureFormat,
    /// Device id, for log lines and error messages.
    device: String,
}

impl CameraSource {
    /// Open `device` with the settings in `config`.
    ///
    /// `device` is parsed as a bare decimal index
    /// ([`DeviceSelector::Index`]) when it is one, and taken as an opaque
    /// platform id ([`DeviceSelector::Id`]) otherwise -- so `"0"` and
    /// `"/dev/video0"` and an AVFoundation `uniqueID` all work, exactly as
    /// they do when handed to the FFmpeg backend.
    ///
    /// # Errors
    ///
    /// [`IoError::Unsupported`] if `config.camera_format` names another
    /// platform's capture API (see [`resolve_camera_platform`]), otherwise
    /// whatever [`map_capture_error`] makes of the open failure.
    pub(super) fn open(config: &VideoConfig, device: &str) -> IoResult<Self> {
        resolve_camera_platform_for_host(config.camera_format.as_deref())?;

        let capture = CaptureConfig {
            device: device_selector(device),
            width: config.target_width.map(capture_dimension).transpose()?,
            height: config.target_height.map(capture_dimension).transpose()?,
            fps: config.camera_fps.map(f64::from),
            // Ordered by what costs least to turn into RGB: NV12 first (one
            // SIMD pass), then the two packed 4:2:2 layouts (one row pass),
            // then planar 4:2:0 (a de-planarised pass) and packed RGB24
            // (a destride). MJPEG is last because it is a full JPEG decode
            // per frame *and* lossy.
            //
            // `Yuv420p` and `Rgb24` are listed explicitly rather than left
            // unlisted: `negotiate` ranks every unlisted encoding *after*
            // every listed one, so omitting them would send a device that
            // offers `{Yuv420p, MJPEG}` down the JPEG path even though the
            // raw layout is right there and cheaper. Encodings still absent
            // from this list (Gray8, the high-bit-depth layouts) remain
            // eligible at the bottom, so a device offering only one of those
            // is opened rather than refused -- `convert_capture_video_frame`
            // handles the ones it can and names the ones it cannot.
            preferred: vec![
                CaptureEncoding::Raw(CapturePixelFormat::Nv12),
                CaptureEncoding::Raw(CapturePixelFormat::Yuyv422),
                CaptureEncoding::Raw(CapturePixelFormat::Uyvy422),
                CaptureEncoding::Raw(CapturePixelFormat::Yuv420p),
                CaptureEncoding::Raw(CapturePixelFormat::Rgb24),
                CaptureEncoding::Mjpeg,
            ],
            allow_compressed: true,
            // The latency budget, in frames. `buffer_size` means the same
            // thing on both sides of this call, and a zero-capacity queue
            // would make the drop policy meaningless.
            queue_depth: config.buffer_size.max(1),
            // The only defensible policy for a live camera: keep the driver
            // ring cycling and bound latency at `queue_depth` frames.
            drop_policy: DropPolicy::DropOldest,
            ..CaptureConfig::default()
        };

        let session =
            oximedia_capture::open(capture).map_err(|error| map_capture_error(error, device))?;
        Self::adopt(session)
    }

    /// Wrap an already-opened [`CaptureSession`].
    ///
    /// Shared by [`CameraSource::open`] and the reader's test seam, so a
    /// scripted mock session travels the exact code path a real device does
    /// from here on.
    ///
    /// # Errors
    ///
    /// [`IoError::ReadFailed`] if the session's stream has already been taken
    /// -- a session has one delivery queue, and a second handle would split
    /// the frames between the two.
    pub(super) fn adopt(mut session: CaptureSession) -> IoResult<Self> {
        let negotiated = session.negotiated_format();
        let device = session.device().id.clone();
        let stream = session.take_stream().ok_or_else(|| {
            IoError::ReadFailed(format!(
                "the capture session for camera '{device}' has already handed out its frame \
                 stream"
            ))
        })?;

        Ok(Self {
            session,
            stream,
            mjpeg: None,
            fps: negotiated.fps(),
            negotiated,
            device,
        })
    }

    /// The mode the device was actually configured for.
    pub(super) fn negotiated(&self) -> CaptureFormat {
        self.negotiated
    }

    /// Frame rate of the negotiated mode, or `0.0` when the device reports a
    /// variable / unknown rate.
    pub(super) fn fps(&self) -> f64 {
        self.fps
    }

    /// Device id, as the backend reports it.
    pub(super) fn device(&self) -> &str {
        &self.device
    }

    /// The capture backend behind this session.
    pub(super) fn backend(&self) -> BackendKind {
        self.session.backend()
    }

    /// Block for the next captured frame, or `Ok(None)` at end of session.
    ///
    /// # Errors
    ///
    /// The backend's terminal error, mapped by [`map_capture_error`]. A
    /// session that ended because of one reports it exactly once and then
    /// reports end-of-stream.
    pub(super) fn recv(&mut self) -> IoResult<Option<CaptureFrame>> {
        let device = self.device.clone();
        self.stream
            .recv()
            .map_err(|error| map_capture_error(error, &device))
    }

    /// Convert one captured frame into packed HWC bytes in `target.format`,
    /// rescaled to `target`'s dimensions.
    ///
    /// # Errors
    ///
    /// * [`IoError::ReadFailed`] if the payload and the negotiated format
    ///   disagree, if a plane is shorter than its geometry demands, or if the
    ///   MJPEG decode fails.
    /// * [`IoError::Unsupported`] for a pixel layout with no conversion here.
    pub(super) fn convert(
        &mut self,
        frame: &CaptureFrame,
        target: FrameTarget,
    ) -> IoResult<Vec<u8>> {
        match &frame.payload {
            FramePayload::Raw(video) => {
                check_raw_agreement(frame.format, video)?;
                convert_capture_video_frame(video, target)
            }
            FramePayload::Compressed(bytes) => {
                if frame.format.encoding != CaptureEncoding::Mjpeg {
                    return Err(IoError::ReadFailed(format!(
                        "camera '{}' delivered a compressed payload while its format says \
                         {}; the pure-Rust backend decodes MJPEG only and will not guess \
                         what the bitstream is",
                        self.device, frame.format.encoding
                    )));
                }
                let video = self.decode_mjpeg(bytes, frame)?;
                convert_capture_video_frame(&video, target)
            }
        }
    }

    /// Decode one MJPEG payload into a raw frame.
    ///
    /// The decoder is built at the negotiated dimensions on first use, but
    /// the geometry the conversion works from comes from the *decoded* frame:
    /// `MjpegDecoder` sizes its output from the JPEG's own SOF0 header, not
    /// from the constructor arguments, so trusting the constructor would
    /// mis-slice any device whose JPEGs disagree with its advertised mode.
    ///
    /// `MjpegDecoder` emits `Rgb24` unless told otherwise, so the common
    /// camera path is JPEG -> RGB24 -> (destride) -> resize, with no second
    /// colour conversion.
    fn decode_mjpeg(&mut self, jpeg: &[u8], frame: &CaptureFrame) -> IoResult<CaptureVideoFrame> {
        let (width, height) = (self.negotiated.width, self.negotiated.height);
        let device = self.device.clone();
        let decoder = self
            .mjpeg
            .get_or_insert_with(|| MjpegDecoder::new(width, height));

        // `MjpegDecoder` stamps the frame with `pts` on a 1/1000 timebase, so
        // milliseconds is the unit it expects. Saturating rather than
        // wrapping: a session cannot run for 2^63 ms, but a corrupt device
        // clock should not produce a negative timestamp.
        let pts = i64::try_from(frame.timestamp.as_millis()).unwrap_or(i64::MAX);
        decoder.send_packet(jpeg, pts).map_err(|error| {
            IoError::ReadFailed(format!(
                "camera '{device}': MJPEG packet rejected ({} bytes): {error}",
                jpeg.len()
            ))
        })?;

        match decoder.receive_frame() {
            Ok(Some(video)) => Ok(video),
            // Every MJPEG packet is one complete image, so the decoder never
            // legitimately holds output back -- `Ok(None)` here means the
            // packet was accepted and then vanished.
            Ok(None) => Err(IoError::ReadFailed(format!(
                "camera '{device}': the MJPEG decoder produced no frame for a complete \
                 JPEG image"
            ))),
            Err(error) => Err(IoError::ReadFailed(format!(
                "camera '{device}': MJPEG decode failed: {error}"
            ))),
        }
    }
}

/// Parse a `VideoSource::Camera` string into a capture device selector.
///
/// A bare decimal number is an index; anything else is an opaque
/// platform-specific id (a `/dev/video*` path, an AVFoundation `uniqueID`, a
/// Media Foundation symbolic link).
fn device_selector(device: &str) -> DeviceSelector {
    let trimmed = device.trim();
    match trimmed.parse::<usize>() {
        Ok(index) => DeviceSelector::Index(index),
        Err(_) => DeviceSelector::Id(trimmed.to_string()),
    }
}

/// Narrow a requested pixel dimension to the `u32` a [`CaptureConfig`] takes.
fn capture_dimension(value: usize) -> IoResult<u32> {
    u32::try_from(value)
        .map_err(|_| IoError::ConfigError(format!("Frame dimension {value} does not fit in u32")))
}

/// Refuse a raw payload whose pixel layout contradicts the negotiated format.
///
/// Conversion dispatches on the frame's own `format` field (the bytes are
/// laid out the way the decoder says, not the way the mode says), but the two
/// disagreeing means one of them is wrong, and guessing which would silently
/// misread every pixel.
fn check_raw_agreement(format: CaptureFormat, video: &CaptureVideoFrame) -> IoResult<()> {
    match format.encoding {
        CaptureEncoding::Raw(declared) if declared == video.format => Ok(()),
        other => Err(IoError::ReadFailed(format!(
            "capture delivered a raw {:?} frame while the negotiated format says {other}; \
             refusing to guess which one describes the bytes",
            video.format
        ))),
    }
}

// ============================================================================
// Pixel conversion
// ============================================================================

/// Convert one decoded capture frame into packed HWC bytes in
/// `target.format`, rescaled to `target`'s dimensions.
///
/// Dispatches on the frame's own [`CapturePixelFormat`] and shares every
/// downstream step (`planar_to_rgb24`, `resize_packed`, `rgb24_to_rgba32`)
/// with the Y4M path in `backend_pure`, so a 4:2:0 camera frame and a 4:2:0
/// Y4M picture that reach the same output format go through the same
/// arithmetic.
///
/// The one deliberate exception is even-sized NV12, which goes through
/// `oximedia-core`'s SIMD `nv12_to_rgb24`. That kernel uses its own
/// fixed-point BT.601 coefficients and is *not* bit-identical to
/// `PixelConverter`'s (a few code values apart on saturated colours) -- which
/// is why nothing here asserts cross-path byte equality the way
/// `backend_pure`'s `upstream_and_generic_420_paths_agree` does for Y4M.
pub(super) fn convert_capture_video_frame(
    video: &CaptureVideoFrame,
    target: FrameTarget,
) -> IoResult<Vec<u8>> {
    let width = usize::try_from(video.width).map_err(|_| {
        IoError::ReadFailed(format!("Capture frame width {} is absurd", video.width))
    })?;
    let height = usize::try_from(video.height).map_err(|_| {
        IoError::ReadFailed(format!("Capture frame height {} is absurd", video.height))
    })?;
    if width == 0 || height == 0 || target.width == 0 || target.height == 0 {
        return Err(IoError::ConfigError(
            "Video frame dimensions must be non-zero".into(),
        ));
    }

    let geometry = PlaneGeometry::packed(width, height);

    match target.format {
        PixelFormat::Gray => {
            let gray = capture_luma(video, width, height)?;
            resize_packed(gray, geometry, target, 1)
        }
        PixelFormat::Rgb | PixelFormat::Rgba => {
            let scaled = if video.format == CapturePixelFormat::Gray8 {
                // Rescale the single plane FIRST, then expand 1 -> 3: the
                // same optimisation `backend_pure::convert_mono` makes, and
                // safe for the same reason (there is no chroma plane that
                // could fall out of step with luma).
                let gray = capture_luma(video, width, height)?;
                let scaled_gray = resize_packed(gray, geometry, target, 1)?;
                gray8_to_rgb24(&scaled_gray, target.width, target.height)
            } else {
                // Subsampled sources must be converted before rescaling --
                // see `backend_pure::convert_planar` for why resizing chroma
                // on its own grid smears colour across edges.
                let rgb = capture_rgb24(video, width, height)?;
                resize_packed(rgb, geometry, target, 3)?
            };
            if matches!(target.format, PixelFormat::Rgba) {
                Ok(rgb24_to_rgba32(&scaled))
            } else {
                Ok(scaled)
            }
        }
    }
}

/// The luma (greyscale) samples of a capture frame, tightly packed.
///
/// Identity for every Y'CbCr layout -- no BT.601 limited-to-full range
/// expansion -- exactly as `backend_pure` treats a Y4M luma plane and as
/// libswscale treats `yuv420p -> GRAY8`. Packed RGB has no luma plane, so it
/// goes through the BT.601 luma weights instead.
fn capture_luma(video: &CaptureVideoFrame, width: usize, height: usize) -> IoResult<Vec<u8>> {
    match video.format {
        CapturePixelFormat::Gray8
        | CapturePixelFormat::Nv12
        | CapturePixelFormat::Nv21
        | CapturePixelFormat::Yuv420p
        | CapturePixelFormat::Yuv422p
        | CapturePixelFormat::Yuv444p => {
            destride(capture_plane(video, 0, "Y")?, width, height, "Y")
        }
        CapturePixelFormat::Yuyv422 => packed_422_luma(video, width, height, 0),
        CapturePixelFormat::Uyvy422 => packed_422_luma(video, width, height, 1),
        CapturePixelFormat::Rgb24 => {
            let row_bytes = checked_row_bytes(width, 3)?;
            let rgb = destride(capture_plane(video, 0, "RGB")?, row_bytes, height, "RGB")?;
            Ok(rgb24_to_gray8(&rgb, width, height))
        }
        other => Err(unsupported_capture_format(other)),
    }
}

/// Packed RGB24 for a capture frame, at the frame's own dimensions.
fn capture_rgb24(video: &CaptureVideoFrame, width: usize, height: usize) -> IoResult<Vec<u8>> {
    match video.format {
        CapturePixelFormat::Rgb24 => {
            let row_bytes = checked_row_bytes(width, 3)?;
            destride(capture_plane(video, 0, "RGB")?, row_bytes, height, "RGB")
        }
        CapturePixelFormat::Gray8 => {
            let gray = destride(capture_plane(video, 0, "Y")?, width, height, "Y")?;
            Ok(gray8_to_rgb24(&gray, width, height))
        }
        CapturePixelFormat::Nv12 => semi_planar_to_rgb24(video, width, height, false),
        CapturePixelFormat::Nv21 => semi_planar_to_rgb24(video, width, height, true),
        CapturePixelFormat::Yuv420p => tri_planar_to_rgb24(video, width, height, 1, 1),
        CapturePixelFormat::Yuv422p => tri_planar_to_rgb24(video, width, height, 1, 0),
        CapturePixelFormat::Yuv444p => tri_planar_to_rgb24(video, width, height, 0, 0),
        CapturePixelFormat::Yuyv422 => packed_422_to_rgb24(video, width, height, false),
        CapturePixelFormat::Uyvy422 => packed_422_to_rgb24(video, width, height, true),
        other => Err(unsupported_capture_format(other)),
    }
}

/// The error for a pixel layout this backend has no conversion for.
fn unsupported_capture_format(format: CapturePixelFormat) -> IoError {
    IoError::Unsupported(format!(
        "the pure-Rust video backend cannot convert captured {format:?} frames: it handles \
         Gray8, Nv12, Nv21, Yuv420p, Yuv422p, Yuv444p, Yuyv422, Uyvy422 and Rgb24 -- enable \
         the `video` (FFmpeg) feature for anything else"
    ))
}

/// Planar Y'CbCr (three separate planes) to packed RGB24.
///
/// Hands straight to `backend_pure`'s shared converter, which takes
/// `oximedia-core`'s 4:2:0 fast path for even dimensions and its own generic
/// loop otherwise.
fn tri_planar_to_rgb24(
    video: &CaptureVideoFrame,
    width: usize,
    height: usize,
    x_shift: u32,
    y_shift: u32,
) -> IoResult<Vec<u8>> {
    let geometry = PlaneGeometry::with_shifts(width, height, x_shift, y_shift);
    let chroma_width = geometry.chroma_width();
    let chroma_height = geometry.chroma_height();

    let luma = destride(capture_plane(video, 0, "Y")?, width, height, "Y")?;
    let u = destride(
        capture_plane(video, 1, "Cb")?,
        chroma_width,
        chroma_height,
        "Cb",
    )?;
    let v = destride(
        capture_plane(video, 2, "Cr")?,
        chroma_width,
        chroma_height,
        "Cr",
    )?;

    planar_to_rgb24(&luma, &u, &v, geometry)
}

/// Semi-planar 4:2:0 (`Nv12` / `Nv21`: luma plane plus one interleaved chroma
/// plane) to packed RGB24.
///
/// Even dimensions take `oximedia-core`'s SIMD `nv12_to_rgb24` (SSE4.1 where
/// the CPU has it, a scalar kernel otherwise). Odd ones cannot: that function
/// indexes chroma as `(row / 2) * width + (col & !1)` and `debug_assert`s a
/// plane length of exactly `(width / 2) * (height / 2) * 2`, both of which are
/// wrong when a dimension is odd and the plane is `ceil(w/2) x ceil(h/2)`. So
/// the odd case de-interleaves the chroma plane and reuses `backend_pure`'s
/// generic planar loop, which was written for exactly that mismatch on the
/// Y4M side.
fn semi_planar_to_rgb24(
    video: &CaptureVideoFrame,
    width: usize,
    height: usize,
    swapped: bool,
) -> IoResult<Vec<u8>> {
    let geometry = PlaneGeometry::with_shifts(width, height, 1, 1);
    let chroma_width = geometry.chroma_width();
    let chroma_height = geometry.chroma_height();
    let chroma_row_bytes = checked_row_bytes(chroma_width, 2)?;

    let luma = destride(capture_plane(video, 0, "Y")?, width, height, "Y")?;
    let chroma = destride(
        capture_plane(video, 1, "CbCr")?,
        chroma_row_bytes,
        chroma_height,
        "CbCr",
    )?;

    if width.is_multiple_of(2) && height.is_multiple_of(2) {
        // Both orders go through `oximedia-core`'s own kernels rather than
        // one taking the fast path and the other the generic loop: the two
        // round differently, and NV12 and NV21 carrying the same picture must
        // decode to the same bytes.
        let rgb = if swapped {
            nv21_to_rgb24(&luma, &chroma, width, height, SimdColorMatrix::Bt601)
        } else {
            nv12_to_rgb24(&luma, &chroma, width, height, SimdColorMatrix::Bt601)
        };
        return Ok(rgb);
    }

    // De-interleave into the two planes the generic loop expects. NV21 is the
    // same wire format with the chroma pair swapped, which is the whole of the
    // difference between the two.
    let samples = chroma_width.saturating_mul(chroma_height);
    let mut u = Vec::with_capacity(samples);
    let mut v = Vec::with_capacity(samples);
    for pair in chroma.chunks_exact(2) {
        if swapped {
            v.push(pair[0]);
            u.push(pair[1]);
        } else {
            u.push(pair[0]);
            v.push(pair[1]);
        }
    }

    planar_to_rgb24_generic(&luma, &u, &v, geometry)
}

/// Packed 4:2:2 (`Yuyv422` / `Uyvy422`) to packed RGB24, one row at a time.
///
/// `oximedia-simd`'s row converters `assert!` (not `debug_assert!`) that the
/// source row holds `ceil(width / 2) * 4` bytes and the destination `width *
/// 3`, so a short row would abort the process in a release build. Every row
/// is therefore sized and bounds-checked here first, and a row that really is
/// short of its final macropixel -- only possible for an odd width, which no
/// real 4:2:2 device produces -- is copied into a scratch buffer padded with
/// neutral chroma (128) rather than being handed over short.
fn packed_422_to_rgb24(
    video: &CaptureVideoFrame,
    width: usize,
    height: usize,
    swapped: bool,
) -> IoResult<Vec<u8>> {
    let plane = capture_plane(video, 0, "YUV 4:2:2")?;
    let sample_bytes = checked_row_bytes(width, 2)?;
    let macropixel_bytes = checked_row_bytes(width.div_ceil(2), 4)?;
    let out_row_bytes = checked_row_bytes(width, 3)?;

    let mut rgb = vec![0u8; out_row_bytes.saturating_mul(height)];
    let mut scratch = vec![128u8; macropixel_bytes];

    for (row, out_row) in rgb.chunks_exact_mut(out_row_bytes).enumerate() {
        let start = row.checked_mul(plane.stride).ok_or_else(|| {
            IoError::ReadFailed("Capture 4:2:2 row offset overflows usize".to_string())
        })?;
        let available = plane
            .data
            .len()
            .saturating_sub(start)
            .min(plane.stride)
            .min(macropixel_bytes);
        if available < sample_bytes {
            return Err(IoError::ReadFailed(format!(
                "Truncated 4:2:2 capture frame: row {row} carries {available} of the \
                 {sample_bytes} bytes a {width}-pixel row needs"
            )));
        }

        let source = plane.data.get(start..start + available).ok_or_else(|| {
            IoError::ReadFailed(format!("Truncated 4:2:2 capture frame at row {row}"))
        })?;
        let source = if available == macropixel_bytes {
            source
        } else {
            // Odd width only: the trailing V byte of the final macropixel is
            // genuinely absent from the wire format, and 128 (neutral chroma)
            // is the least wrong stand-in. The rest of `scratch` is rewritten
            // every row, and `macropixel_bytes - available` is at most 2.
            scratch[..available].copy_from_slice(source);
            &scratch[..]
        };

        if swapped {
            convert_uyvy422_row_to_rgb(source, out_row, width);
        } else {
            convert_yuv422_row_to_rgb(source, out_row, width);
        }
    }

    Ok(rgb)
}

/// The luma samples of a packed 4:2:2 frame.
///
/// `offset` is where the first Y byte sits inside a macropixel: 0 for YUYV,
/// 1 for UYVY. Luma is every other byte from there.
fn packed_422_luma(
    video: &CaptureVideoFrame,
    width: usize,
    height: usize,
    offset: usize,
) -> IoResult<Vec<u8>> {
    let plane = capture_plane(video, 0, "YUV 4:2:2")?;
    let sample_bytes = checked_row_bytes(width, 2)?;
    let mut gray = vec![0u8; width.saturating_mul(height)];

    for (row, out_row) in gray.chunks_exact_mut(width.max(1)).enumerate() {
        let start = row.checked_mul(plane.stride).ok_or_else(|| {
            IoError::ReadFailed("Capture 4:2:2 row offset overflows usize".to_string())
        })?;
        let end = start.checked_add(sample_bytes).ok_or_else(|| {
            IoError::ReadFailed("Capture 4:2:2 row length overflows usize".to_string())
        })?;
        let source = plane.data.get(start..end).ok_or_else(|| {
            IoError::ReadFailed(format!(
                "Truncated 4:2:2 capture frame: row {row} needs bytes {start}..{end} of a \
                 {}-byte plane",
                plane.data.len()
            ))
        })?;
        for (column, sample) in out_row.iter_mut().enumerate() {
            // `column < width` and `source` holds `width * 2` bytes, so
            // `column * 2 + offset` is at most `2 * width - 1`: in bounds.
            *sample = source[column * 2 + offset];
        }
    }

    Ok(gray)
}

/// Borrow plane `index` of a capture frame, or report the frame as malformed.
fn capture_plane<'a>(
    video: &'a CaptureVideoFrame,
    index: usize,
    label: &str,
) -> IoResult<&'a Plane> {
    video.planes.get(index).ok_or_else(|| {
        IoError::ReadFailed(format!(
            "Captured {:?} frame has no {label} plane (index {index} of {})",
            video.format,
            video.planes.len()
        ))
    })
}

/// `width * bytes_per_sample`, or an error rather than a wrap.
fn checked_row_bytes(width: usize, bytes_per_sample: usize) -> IoResult<usize> {
    width.checked_mul(bytes_per_sample).ok_or_else(|| {
        IoError::ConfigError(format!(
            "A row of {width} pixels at {bytes_per_sample} bytes each overflows usize"
        ))
    })
}

/// Copy `rows` rows of `row_bytes` bytes out of a strided plane into a
/// tightly packed buffer.
///
/// Capture backends hand back driver buffers whose stride is padded to a
/// hardware alignment (64 bytes on many V4L2 drivers, 16 or 64 on
/// AVFoundation), and every converter downstream of here assumes
/// `stride == row_bytes`. Copying is what removes that assumption -- and when
/// the plane is already tight the copy is a single `to_vec`, not a per-row
/// loop.
///
/// # Errors
///
/// [`IoError::ReadFailed`] if the stride is narrower than a row (the plane
/// cannot hold the picture it claims) or the buffer ends before the last row.
fn destride(plane: &Plane, row_bytes: usize, rows: usize, label: &str) -> IoResult<Vec<u8>> {
    if row_bytes == 0 || rows == 0 {
        return Ok(Vec::new());
    }
    if plane.stride < row_bytes {
        return Err(IoError::ReadFailed(format!(
            "Capture {label} plane stride {} is narrower than its {row_bytes}-byte rows",
            plane.stride
        )));
    }

    let total = row_bytes.checked_mul(rows).ok_or_else(|| {
        IoError::ReadFailed(format!("Capture {label} plane size overflows usize"))
    })?;

    if plane.stride == row_bytes {
        return plane
            .data
            .get(..total)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| truncated_plane(label, total, plane.data.len()));
    }

    let mut packed = Vec::with_capacity(total);
    for row in 0..rows {
        let start = row.checked_mul(plane.stride).ok_or_else(|| {
            IoError::ReadFailed(format!("Capture {label} plane row offset overflows usize"))
        })?;
        let end = start.checked_add(row_bytes).ok_or_else(|| {
            IoError::ReadFailed(format!("Capture {label} plane row length overflows usize"))
        })?;
        let source = plane
            .data
            .get(start..end)
            .ok_or_else(|| truncated_plane(label, end, plane.data.len()))?;
        packed.extend_from_slice(source);
    }

    Ok(packed)
}

/// The error for a plane buffer that ends before its geometry says it should.
fn truncated_plane(label: &str, needed: usize, have: usize) -> IoError {
    IoError::ReadFailed(format!(
        "Truncated capture frame: the {label} plane needs {needed} bytes, the buffer holds \
         {have}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(format: CapturePixelFormat, width: u32, height: u32) -> CaptureVideoFrame {
        let mut video = CaptureVideoFrame::new(format, width, height);
        video.allocate();
        video
    }

    /// A semi-planar (`Nv12` / `Nv21`) frame whose interleaved chroma plane is
    /// the size the layout actually requires.
    ///
    /// Built by hand for explicit control over fill values. (Historical note:
    /// `oximedia_codec::VideoFrame::allocate` used to under-size the NV12
    /// chroma plane 2x by deriving it from chroma *pixel* dimensions; that is
    /// fixed upstream, so `allocate` now agrees with this helper and with the
    /// real capture backends, which always built their planes from
    /// `PixelFormat::stride_for_width` -- see `platform/macos/pixel.rs` and
    /// `platform/mf_logic.rs` in `oximedia-capture`.)
    fn semi_planar_frame(
        format: CapturePixelFormat,
        width: u32,
        height: u32,
        luma: u8,
        chroma: u8,
    ) -> CaptureVideoFrame {
        let mut video = CaptureVideoFrame::new(format, width, height);
        let (w, h) = (width as usize, height as usize);
        let chroma_width = w.div_ceil(2);
        let chroma_height = h.div_ceil(2);
        video.planes = vec![
            Plane::with_dimensions(vec![luma; w * h], w, width, height),
            Plane::with_dimensions(
                vec![chroma; chroma_width * 2 * chroma_height],
                chroma_width * 2,
                chroma_width as u32,
                chroma_height as u32,
            ),
        ];
        video
    }

    fn rgb_target(width: usize, height: usize) -> FrameTarget {
        FrameTarget::new(PixelFormat::Rgb, width, height)
    }

    // ── Alias table ─────────────────────────────────────────────────────────

    /// Every accepted row resolves to the backend it names, when that backend
    /// is the host's.
    #[test]
    fn every_alias_resolves_to_its_backend_on_its_own_host() {
        for (alias, expected) in CAMERA_FORMAT_ALIASES {
            let resolved = resolve_camera_platform(Some(alias), *expected);
            match resolved {
                Ok(kind) => assert_eq!(kind, *expected, "alias {alias}"),
                Err(error) => panic!("alias {alias} must resolve on its own host: {error}"),
            }
        }
    }

    /// The Windows rows: DirectShow and VfW are served through Media
    /// Foundation rather than refused, and `msmf` names it directly.
    #[test]
    fn legacy_windows_aliases_map_to_media_foundation() {
        for alias in ["dshow", "vfwcap", "msmf"] {
            assert_eq!(
                resolve_camera_platform(Some(alias), BackendKind::MediaFoundation)
                    .expect("legacy Windows alias"),
                BackendKind::MediaFoundation,
                "alias {alias}"
            );
        }
    }

    /// Alias matching ignores case and surrounding whitespace, because
    /// `camera_format` is a free-form user string.
    #[test]
    fn alias_matching_is_case_and_whitespace_insensitive() {
        assert_eq!(
            resolve_camera_platform(Some("  AVFoundation "), BackendKind::AvFoundation)
                .expect("case-insensitive alias"),
            BackendKind::AvFoundation
        );
    }

    /// `auto`, `None` and an empty string all mean "this host's backend".
    #[test]
    fn auto_and_absent_formats_take_the_host_backend() {
        for name in [None, Some("auto"), Some("AUTO"), Some("   "), Some("")] {
            assert_eq!(
                resolve_camera_platform(name, BackendKind::V4l2).expect("host default"),
                BackendKind::V4l2,
                "name {name:?}"
            );
        }
    }

    /// An unknown name is an error that lists what *is* accepted, rather than
    /// a silent fallback to the host backend.
    #[test]
    fn unknown_format_names_the_accepted_aliases() {
        let error = match resolve_camera_platform(Some("gdigrab"), BackendKind::V4l2) {
            Err(error) => error,
            Ok(kind) => panic!("'gdigrab' is not a capture backend, got {kind}"),
        };
        assert!(matches!(error, IoError::Unsupported(_)), "{error:?}");
        let message = error.to_string();
        for alias in ["video4linux2", "avfoundation", "dshow", "msmf", "auto"] {
            assert!(message.contains(alias), "{message} must list {alias}");
        }
    }

    /// A name that selects another platform's capture API is an error, never
    /// a silent substitution of the host's.
    #[test]
    fn a_foreign_backend_is_refused_not_substituted() {
        let cases = [
            ("v4l2", BackendKind::AvFoundation),
            ("video4linux2", BackendKind::MediaFoundation),
            ("avfoundation", BackendKind::V4l2),
            ("dshow", BackendKind::V4l2),
            ("msmf", BackendKind::AvFoundation),
            ("vfwcap", BackendKind::AvFoundation),
        ];
        for (name, host) in cases {
            let error = match resolve_camera_platform(Some(name), host) {
                Err(error) => error,
                Ok(kind) => panic!("'{name}' must not resolve on a {host} host (got {kind})"),
            };
            assert!(
                matches!(error, IoError::Unsupported(_)),
                "{name}: {error:?}"
            );
            assert!(
                error.to_string().contains("never substitutes"),
                "{name}: {error}"
            );
        }
    }

    /// A target with no capture backend compiled in says so, rather than
    /// resolving to `Unsupported` and failing later at open time.
    #[test]
    fn a_host_without_a_backend_is_refused_up_front() {
        for name in [None, Some("auto"), Some("v4l2")] {
            let error = match resolve_camera_platform(name, BackendKind::Unsupported) {
                Err(error) => error,
                Ok(kind) => panic!("{name:?} must not resolve without a backend (got {kind})"),
            };
            assert!(matches!(error, IoError::Unsupported(_)), "{name:?}");
            assert!(
                error.to_string().contains("oximedia-capture implements"),
                "{name:?}: {error}"
            );
        }
    }

    /// The mock backend is a real backend as far as availability goes, so a
    /// mock-driven session is never rejected by the platform check.
    #[test]
    fn the_mock_backend_counts_as_available() {
        assert!(BackendKind::Mock.is_available());
    }

    // ── Device selector ─────────────────────────────────────────────────────

    #[test]
    fn bare_numbers_are_indices_and_everything_else_is_an_id() {
        assert_eq!(device_selector("0"), DeviceSelector::Index(0));
        assert_eq!(device_selector(" 12 "), DeviceSelector::Index(12));
        assert_eq!(
            device_selector("/dev/video0"),
            DeviceSelector::Id("/dev/video0".to_string())
        );
        assert_eq!(
            device_selector("0x8020000005ac8514"),
            DeviceSelector::Id("0x8020000005ac8514".to_string())
        );
        // A negative number is not an index; it is an opaque id that happens
        // to look numeric.
        assert_eq!(device_selector("-1"), DeviceSelector::Id("-1".to_string()));
    }

    // ── Error mapping ───────────────────────────────────────────────────────

    #[test]
    fn capture_errors_map_to_the_documented_io_error_kinds() {
        let unsupported =
            map_capture_error(CaptureError::UnsupportedPlatform { target: "plan9" }, "cam");
        assert!(matches!(unsupported, IoError::Unsupported(_)));

        let missing = map_capture_error(
            CaptureError::DeviceNotFound(DeviceSelector::Index(7)),
            "cam",
        );
        assert!(matches!(missing, IoError::Connection(_)));
        assert!(missing.to_string().contains('7'), "{missing}");

        let refused = map_capture_error(
            CaptureError::PermissionDenied {
                device: "cam".into(),
            },
            "cam",
        );
        assert!(matches!(refused, IoError::Connection(_)));

        let unsatisfiable = map_capture_error(
            CaptureError::NoMatchingFormat {
                device: "cam".into(),
                available: 3,
            },
            "cam",
        );
        assert!(matches!(unsatisfiable, IoError::ConfigError(_)));
        assert!(unsatisfiable.to_string().contains('3'), "{unsatisfiable}");

        let rejected = map_capture_error(
            CaptureError::format_rejected(
                "cam",
                CaptureFormat::new(CaptureEncoding::Mjpeg, 1920, 1080, 60, 1),
                "not at 60 fps",
            ),
            "cam",
        );
        assert!(matches!(rejected, IoError::ConfigError(_)));

        let io = map_capture_error(
            CaptureError::io(
                "cam",
                std::io::Error::new(std::io::ErrorKind::BrokenPipe, "gone"),
            ),
            "cam",
        );
        assert!(matches!(io, IoError::ReadFailed(_)));

        assert!(matches!(
            map_capture_error(CaptureError::platform("boom"), "cam"),
            IoError::ReadFailed(_)
        ));
        assert!(matches!(
            map_capture_error(CaptureError::ThreadPanic, "cam"),
            IoError::ReadFailed(_)
        ));
        assert!(matches!(
            map_capture_error(CaptureError::Core(oximedia_core::OxiError::Eof), "cam"),
            IoError::ReadFailed(_)
        ));
    }

    /// The macOS message has to name TCC, because that is the only thing the
    /// user can act on.
    #[test]
    #[cfg(target_os = "macos")]
    fn the_macos_permission_message_names_tcc() {
        let error = map_capture_error(
            CaptureError::PermissionDenied {
                device: "cam".into(),
            },
            "cam",
        );
        let message = error.to_string();
        assert!(message.contains("TCC"), "{message}");
        assert!(message.contains("Privacy"), "{message}");
    }

    // ── Destriding ──────────────────────────────────────────────────────────

    #[test]
    fn a_padded_plane_is_copied_row_by_row() {
        // 3 rows of 2 useful bytes inside a 4-byte stride.
        let plane = Plane::with_dimensions(vec![1, 2, 9, 9, 3, 4, 9, 9, 5, 6, 9, 9], 4, 2, 3);
        assert_eq!(
            destride(&plane, 2, 3, "Y").expect("padded plane"),
            vec![1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn a_tight_plane_is_copied_whole() {
        let plane = Plane::with_dimensions(vec![1, 2, 3, 4, 5, 6], 2, 2, 3);
        assert_eq!(
            destride(&plane, 2, 3, "Y").expect("tight plane"),
            vec![1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn a_short_plane_errors_instead_of_panicking() {
        let plane = Plane::with_dimensions(vec![1, 2, 3], 2, 2, 3);
        assert!(destride(&plane, 2, 3, "Y").is_err());
    }

    #[test]
    fn a_stride_narrower_than_a_row_errors() {
        let plane = Plane::with_dimensions(vec![0; 12], 1, 2, 3);
        assert!(destride(&plane, 2, 3, "Y").is_err());
    }

    // ── Conversion ──────────────────────────────────────────────────────────

    #[test]
    fn gray8_converts_to_every_output_format() {
        let mut video = frame(CapturePixelFormat::Gray8, 4, 2);
        if let Some(plane) = video.planes.first_mut() {
            plane.data = (0..8u8).map(|i| i * 30).collect();
        }

        let gray = convert_capture_video_frame(&video, FrameTarget::new(PixelFormat::Gray, 4, 2))
            .expect("gray");
        assert_eq!(gray, (0..8u8).map(|i| i * 30).collect::<Vec<_>>());

        let rgb = convert_capture_video_frame(&video, rgb_target(4, 2)).expect("rgb");
        assert_eq!(rgb.len(), 4 * 2 * 3);
        for (pixel, luma) in rgb.chunks_exact(3).zip(gray.iter()) {
            assert_eq!(
                pixel,
                &[*luma, *luma, *luma],
                "Gray8 -> RGB replicates luma"
            );
        }

        let rgba = convert_capture_video_frame(&video, FrameTarget::new(PixelFormat::Rgba, 4, 2))
            .expect("rgba");
        assert_eq!(rgba.len(), 4 * 2 * 4);
        for pixel in rgba.chunks_exact(4) {
            assert_eq!(pixel[3], 255, "captured frames carry no alpha");
        }
    }

    #[test]
    fn rgb24_passes_through_and_destrides() {
        let mut video = CaptureVideoFrame::new(CapturePixelFormat::Rgb24, 2, 2);
        // Stride padded past the 6 useful bytes per row, as a driver buffer is.
        video.planes = vec![Plane::with_dimensions(
            vec![1, 2, 3, 4, 5, 6, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0],
            8,
            2,
            2,
        )];
        let rgb = convert_capture_video_frame(&video, rgb_target(2, 2)).expect("rgb24");
        assert_eq!(rgb, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    /// Neutral chroma must decode achromatic through every YUV layout -- the
    /// invariant that holds for any correct converter, and the one worth
    /// asserting given the SIMD and scalar kernels round differently.
    #[test]
    fn neutral_chroma_decodes_achromatic_for_every_yuv_layout() {
        let mut frames = Vec::new();
        for format in [CapturePixelFormat::Nv12, CapturePixelFormat::Nv21] {
            frames.push((format, semi_planar_frame(format, 4, 4, 128, 128)));
        }
        for format in [
            CapturePixelFormat::Yuv420p,
            CapturePixelFormat::Yuv422p,
            CapturePixelFormat::Yuv444p,
        ] {
            let mut video = frame(format, 4, 4);
            for plane in &mut video.planes {
                plane.data.fill(128);
            }
            frames.push((format, video));
        }

        for (format, video) in frames {
            let rgb =
                convert_capture_video_frame(&video, rgb_target(4, 4)).expect("neutral chroma");
            assert_eq!(rgb.len(), 4 * 4 * 3, "{format:?}");
            for pixel in rgb.chunks_exact(3) {
                assert!(
                    pixel[0] == pixel[1] && pixel[1] == pixel[2],
                    "{format:?}: neutral chroma must be achromatic, got {pixel:?}"
                );
            }
        }
    }

    /// A chroma plane that is too small for the layout it claims must be an
    /// error, not a half-read.
    ///
    /// `oximedia_codec::VideoFrame::allocate` used to produce exactly this
    /// malformed shape for NV12 (chroma sized from pixel dimensions instead
    /// of interleaved byte width); that bug is fixed upstream, so the frame
    /// is now built by hand to keep guarding the refusal path against any
    /// other producer that hands over a half-width chroma plane.
    #[test]
    fn an_undersized_semi_planar_chroma_plane_is_refused() {
        let mut video = CaptureVideoFrame::new(CapturePixelFormat::Nv12, 64, 48);
        let (w, h) = (64usize, 48usize);
        // Luma is correct; chroma is the half-width mistake (32-byte rows
        // where the interleaved U/V layout needs 64).
        video.planes = vec![
            Plane::with_dimensions(vec![0u8; w * h], w, 64, 48),
            Plane::with_dimensions(vec![128u8; (w / 2) * (h / 2)], w / 2, 32, 24),
        ];
        let error = match convert_capture_video_frame(&video, rgb_target(64, 48)) {
            Err(error) => error,
            Ok(_) => panic!("a half-sized NV12 chroma plane must not be read as a whole one"),
        };
        assert!(matches!(error, IoError::ReadFailed(_)), "{error:?}");
        assert!(error.to_string().contains("narrower"), "{error}");
    }

    /// Packed 4:2:2 in both byte orders: the luma path picks the right byte,
    /// and the RGB path produces a full-size picture.
    #[test]
    fn packed_422_reads_both_byte_orders() {
        // One 2-pixel macropixel per row, two rows.
        let yuyv = vec![40, 128, 60, 128, 80, 128, 100, 128];
        let mut video = CaptureVideoFrame::new(CapturePixelFormat::Yuyv422, 2, 2);
        video.planes = vec![Plane::with_dimensions(yuyv.clone(), 4, 2, 2)];
        let gray = convert_capture_video_frame(&video, FrameTarget::new(PixelFormat::Gray, 2, 2))
            .expect("yuyv luma");
        assert_eq!(gray, vec![40, 60, 80, 100]);
        assert_eq!(
            convert_capture_video_frame(&video, rgb_target(2, 2))
                .expect("yuyv rgb")
                .len(),
            2 * 2 * 3
        );

        // UYVY is the same data with the pairs swapped.
        let uyvy = vec![128, 40, 128, 60, 128, 80, 128, 100];
        let mut video = CaptureVideoFrame::new(CapturePixelFormat::Uyvy422, 2, 2);
        video.planes = vec![Plane::with_dimensions(uyvy, 4, 2, 2)];
        let gray = convert_capture_video_frame(&video, FrameTarget::new(PixelFormat::Gray, 2, 2))
            .expect("uyvy luma");
        assert_eq!(gray, vec![40, 60, 80, 100]);
    }

    /// An odd-width 4:2:2 row is short of its final macropixel; the upstream
    /// converter `assert!`s on that, so this path must pad rather than panic.
    #[test]
    fn odd_width_422_pads_instead_of_panicking() {
        // 3 pixels: 6 sample bytes, but the converters want 2 macropixels = 8.
        let mut video = CaptureVideoFrame::new(CapturePixelFormat::Yuyv422, 3, 1);
        video.planes = vec![Plane::with_dimensions(
            vec![40, 128, 60, 128, 80, 128],
            6,
            3,
            1,
        )];
        let rgb = convert_capture_video_frame(&video, rgb_target(3, 1)).expect("odd 4:2:2");
        assert_eq!(rgb.len(), 3 * 3);
    }

    /// Odd-sized NV12 must not reach `nv12_to_rgb24`, whose scalar indexing
    /// (`(row / 2) * width + (col & !1)`) and `debug_assert`ed plane sizes
    /// both assume even dimensions -- the debug assert would abort a test
    /// build and the indexing would read past the chroma plane in release.
    #[test]
    fn odd_sized_nv12_converts_without_panicking() {
        let video = semi_planar_frame(CapturePixelFormat::Nv12, 3, 3, 128, 128);
        let rgb = convert_capture_video_frame(&video, rgb_target(3, 3)).expect("odd nv12");
        assert_eq!(rgb.len(), 3 * 3 * 3);
        for pixel in rgb.chunks_exact(3) {
            assert!(
                pixel[0] == pixel[1] && pixel[1] == pixel[2],
                "odd NV12 with neutral chroma must stay achromatic, got {pixel:?}"
            );
        }
    }

    /// Resizing goes through the same bilinear resampler the Y4M path uses.
    #[test]
    fn conversion_rescales_to_the_target() {
        let video = semi_planar_frame(CapturePixelFormat::Nv12, 8, 8, 128, 128);
        let rgb = convert_capture_video_frame(&video, rgb_target(4, 2)).expect("resize");
        assert_eq!(rgb.len(), 4 * 2 * 3);
    }

    /// A plane shorter than its geometry is an error, never a panic.
    #[test]
    fn a_truncated_capture_frame_errors() {
        let mut video = CaptureVideoFrame::new(CapturePixelFormat::Nv12, 4, 4);
        video.planes = vec![
            Plane::with_dimensions(vec![0; 8], 4, 4, 4),
            Plane::with_dimensions(vec![0; 8], 4, 2, 2),
        ];
        assert!(convert_capture_video_frame(&video, rgb_target(4, 4)).is_err());
    }

    /// A frame with no planes at all is an error, never an index panic.
    #[test]
    fn a_planeless_capture_frame_errors() {
        let video = CaptureVideoFrame::new(CapturePixelFormat::Nv12, 4, 4);
        assert!(convert_capture_video_frame(&video, rgb_target(4, 4)).is_err());
    }

    /// A layout with no conversion here is refused by name, not misread.
    #[test]
    fn an_unconvertible_layout_is_refused_by_name() {
        let video = frame(CapturePixelFormat::P010, 4, 4);
        let error = match convert_capture_video_frame(&video, rgb_target(4, 4)) {
            Err(error) => error,
            Ok(_) => panic!("P010 has no conversion here"),
        };
        assert!(matches!(error, IoError::Unsupported(_)), "{error:?}");
        assert!(error.to_string().contains("P010"), "{error}");
    }

    /// Zero dimensions are a config error rather than an empty frame.
    #[test]
    fn zero_dimensions_are_rejected() {
        let video = frame(CapturePixelFormat::Gray8, 4, 4);
        assert!(convert_capture_video_frame(&video, rgb_target(0, 4)).is_err());
        let empty = CaptureVideoFrame::new(CapturePixelFormat::Gray8, 0, 4);
        assert!(convert_capture_video_frame(&empty, rgb_target(4, 4)).is_err());
    }

    // ── Payload / format agreement ──────────────────────────────────────────

    // ── MJPEG ───────────────────────────────────────────────────────────────

    /// Encode a genuine baseline JPEG with `oximedia-codec`'s own MJPEG
    /// encoder.
    ///
    /// The capture `mock` backend deliberately refuses to synthesize a
    /// compressed payload -- it "never pretends to be hardware", and emitting
    /// bytes that claim to be a JPEG and are not would be exactly that -- so
    /// the compressed path is exercised against a real bitstream produced
    /// here instead. That means these tests cover the same decode the wire
    /// path takes; what they cannot cover is a *device* handing over the
    /// bytes, which needs hardware.
    fn encode_jpeg(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
        use oximedia_codec::{MjpegConfig, MjpegEncoder, VideoEncoder};

        let config = MjpegConfig::new(width, height)
            .expect("valid MJPEG config")
            .with_quality(95)
            .with_pixel_format(CapturePixelFormat::Rgb24);
        let mut encoder = MjpegEncoder::new(config).expect("valid MJPEG encoder");

        let mut source = CaptureVideoFrame::new(CapturePixelFormat::Rgb24, width, height);
        source.planes = vec![Plane::with_dimensions(
            rgb.to_vec(),
            (width * 3) as usize,
            width,
            height,
        )];

        encoder.send_frame(&source).expect("encode one frame");
        encoder
            .receive_packet()
            .expect("no encoder failure")
            .expect("one packet per frame")
            .data
    }

    /// A red/green/blue/white RGB24 test picture, `width * height` pixels.
    fn rgb_test_picture(width: usize, height: usize) -> Vec<u8> {
        let palette = [
            [220u8, 20, 20],
            [20, 220, 20],
            [20, 20, 220],
            [230, 230, 230],
        ];
        let mut rgb = Vec::with_capacity(width * height * 3);
        for row in 0..height {
            for column in 0..width {
                rgb.extend_from_slice(&palette[(row / 4 * 2 + column / 4) % palette.len()]);
            }
        }
        rgb
    }

    /// A real JPEG bitstream decodes through the camera path and comes out at
    /// the JPEG's own geometry in the configured output format.
    #[test]
    fn a_real_jpeg_decodes_through_the_camera_path() {
        use oximedia_capture::{mock, CaptureConfig};

        let (width, height) = (16usize, 16usize);
        let jpeg = encode_jpeg(
            width as u32,
            height as u32,
            &rgb_test_picture(width, height),
        );
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "a JPEG starts with SOI");

        // The mock cannot negotiate MJPEG (it refuses to fabricate a
        // bitstream), so the session is opened on its raw default and the
        // compressed frame is handed to `convert` directly -- which is the
        // same call `pump_camera` makes.
        let session = mock::open(
            CaptureConfig::default(),
            mock::MockScript::default().with_frames(0),
        )
        .expect("the mock device always exists");
        let mut camera = CameraSource::adopt(session).expect("adopt the mock session");

        let frame = CaptureFrame {
            format: CaptureFormat::new(CaptureEncoding::Mjpeg, width as u32, height as u32, 30, 1),
            payload: FramePayload::Compressed(jpeg.clone()),
            timestamp: std::time::Duration::from_millis(33),
            host_timestamp: std::time::Duration::from_millis(33),
            timestamp_source: oximedia_capture::TimestampSource::DeviceMonotonic,
            sequence: 1,
        };

        let rgb = camera
            .convert(&frame, rgb_target(width, height))
            .expect("decode a real JPEG");
        assert_eq!(rgb.len(), width * height * 3);
        // JPEG is lossy, so nothing here pins exact bytes -- but a decode
        // that produced a flat picture would mean the payload never reached
        // the decoder.
        assert!(
            rgb.iter().any(|&byte| byte != rgb[0]),
            "the decoded picture must carry the encoded pattern"
        );

        // The lazily built decoder is reused for the next frame, and the
        // other output formats work off the same decode.
        let gray = camera
            .convert(&frame, FrameTarget::new(PixelFormat::Gray, width, height))
            .expect("decode again, as Gray");
        assert_eq!(gray.len(), width * height);

        let rgba = camera
            .convert(&frame, FrameTarget::new(PixelFormat::Rgba, 8, 8))
            .expect("decode again, as rescaled Rgba");
        assert_eq!(rgba.len(), 8 * 8 * 4);
        for pixel in rgba.chunks_exact(4) {
            assert_eq!(pixel[3], 255);
        }
    }

    /// Bytes that are not a JPEG are an error, not a frame of garbage.
    #[test]
    fn a_corrupt_mjpeg_payload_is_an_error() {
        use oximedia_capture::{mock, CaptureConfig};

        let session = mock::open(
            CaptureConfig::default(),
            mock::MockScript::default().with_frames(0),
        )
        .expect("the mock device always exists");
        let mut camera = CameraSource::adopt(session).expect("adopt the mock session");

        let frame = CaptureFrame {
            format: CaptureFormat::new(CaptureEncoding::Mjpeg, 8, 8, 30, 1),
            payload: FramePayload::Compressed(vec![0x00; 64]),
            timestamp: std::time::Duration::ZERO,
            host_timestamp: std::time::Duration::ZERO,
            timestamp_source: oximedia_capture::TimestampSource::HostArrival,
            sequence: 0,
        };

        let error = match camera.convert(&frame, rgb_target(8, 8)) {
            Err(error) => error,
            Ok(_) => panic!("64 zero bytes are not a JPEG"),
        };
        assert!(matches!(error, IoError::ReadFailed(_)), "{error:?}");
        assert!(error.to_string().contains("MJPEG"), "{error}");
    }

    /// A compressed payload under a raw negotiated format is a contradiction,
    /// and refused rather than fed to a JPEG decoder on spec.
    #[test]
    fn a_compressed_payload_under_a_raw_format_is_refused() {
        use oximedia_capture::{mock, CaptureConfig};

        let session = mock::open(
            CaptureConfig::default(),
            mock::MockScript::default().with_frames(0),
        )
        .expect("the mock device always exists");
        let mut camera = CameraSource::adopt(session).expect("adopt the mock session");

        let frame = CaptureFrame {
            format: CaptureFormat::new(CaptureEncoding::Raw(CapturePixelFormat::Nv12), 8, 8, 30, 1),
            payload: FramePayload::Compressed(vec![0xFF, 0xD8, 0xFF, 0xD9]),
            timestamp: std::time::Duration::ZERO,
            host_timestamp: std::time::Duration::ZERO,
            timestamp_source: oximedia_capture::TimestampSource::HostArrival,
            sequence: 0,
        };

        let error = match camera.convert(&frame, rgb_target(8, 8)) {
            Err(error) => error,
            Ok(_) => panic!("a compressed payload under a raw format must not decode"),
        };
        assert!(matches!(error, IoError::ReadFailed(_)), "{error:?}");
        assert!(error.to_string().contains("will not guess"), "{error}");
    }

    #[test]
    fn a_payload_that_contradicts_the_negotiated_format_is_refused() {
        let video = frame(CapturePixelFormat::Nv12, 4, 4);
        // The mode says YUYV, the buffer says NV12: one of them is wrong.
        let format = CaptureFormat::new(
            CaptureEncoding::Raw(CapturePixelFormat::Yuyv422),
            4,
            4,
            30,
            1,
        );
        assert!(check_raw_agreement(format, &video).is_err());

        // A compressed mode with a raw payload is the same fault.
        let mjpeg = CaptureFormat::new(CaptureEncoding::Mjpeg, 4, 4, 30, 1);
        assert!(check_raw_agreement(mjpeg, &video).is_err());

        // Agreement passes.
        let agreed =
            CaptureFormat::new(CaptureEncoding::Raw(CapturePixelFormat::Nv12), 4, 4, 30, 1);
        assert!(check_raw_agreement(agreed, &video).is_ok());
    }
}
