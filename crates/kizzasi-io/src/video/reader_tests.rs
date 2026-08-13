//! Integration tests: [`VideoReader`] against real (synthetically generated)
//! Y4M files.
//!
//! Y4M needs no binary test asset -- the file is written byte by byte by
//! [`write_y4m`] below -- and both backends can read it: FFmpeg through its
//! built-in `yuv4mpegpipe` demuxer, the pure-Rust stack through
//! `oximedia-container`'s `Y4mDemuxer`. That is what makes it the parity
//! fixture: the four reader tests below run once per compiled-in backend
//! ([`enabled_backends`]) with *identical* assertions, so a behavioural
//! difference between the two fails the suite instead of hiding behind
//! whichever backend `Auto` happened to pick.
//!
//! VideoReader really opens and decodes its source (id=26 / id=347).

use super::*;

/// Every backend this build can actually decode with.
///
/// Each of the four reader tests loops over this and sets
/// `.with_backend(b)`, so adding a backend automatically extends the parity
/// contract to it.
fn enabled_backends() -> Vec<VideoBackend> {
    // `cfg!` rather than `#[cfg]`: both variants exist in every build (they
    // are plain enum variants), only their backends are feature-gated, and
    // this way the list has one shape for clippy to look at.
    let mut backends = Vec::new();
    if cfg!(feature = "video") {
        backends.push(VideoBackend::Ffmpeg);
    }
    if cfg!(feature = "video-pure") {
        backends.push(VideoBackend::Pure);
    }
    backends
}

/// Write a minimal single-plane (`Cmono`) YUV4MPEG2 file: a raw,
/// dependency-free container that both backends demux without a binary
/// asset.
fn write_y4m(path: &std::path::Path, width: usize, height: usize, frames: usize) {
    use std::io::Write;
    let mut file = std::fs::File::create(path).expect("create y4m");
    writeln!(file, "YUV4MPEG2 W{width} H{height} F25:1 Ip A1:1 Cmono").expect("write y4m header");
    for f in 0..frames {
        file.write_all(b"FRAME\n").expect("write frame header");
        let plane: Vec<u8> = (0..width * height)
            .map(|i| ((i + f * 7) % 200) as u8)
            .collect();
        file.write_all(&plane).expect("write plane");
    }
    file.flush().expect("flush y4m");
}

fn unique_temp_path(tag: &str, extension: &str) -> std::path::PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "kizzasi_io_video_{tag}_{}_{unique}.{extension}",
        std::process::id()
    ))
}

fn temp_y4m(tag: &str, width: usize, height: usize, frames: usize) -> std::path::PathBuf {
    let path = unique_temp_path(tag, "y4m");
    write_y4m(&path, width, height, frames);
    path
}

/// Write a single-frame 4:2:0 (`C420jpeg`) YUV4MPEG2 file with `luma` as the
/// Y plane and neutral (128) chroma, so a correct YUV -> RGB conversion must
/// come out achromatic (R == G == B) and a Gray conversion must hand back
/// `luma` byte for byte.
#[cfg(feature = "video-pure")]
fn temp_y4m_420(tag: &str, width: usize, height: usize, luma: &[u8]) -> std::path::PathBuf {
    use std::io::Write;
    assert_eq!(luma.len(), width * height, "luma plane size");
    assert!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "4:2:0 needs even sides"
    );

    let path = unique_temp_path(tag, "y4m");
    let mut file = std::fs::File::create(&path).expect("create y4m");
    writeln!(file, "YUV4MPEG2 W{width} H{height} F25:1 Ip A1:1 C420jpeg")
        .expect("write y4m header");
    file.write_all(b"FRAME\n").expect("write frame header");
    file.write_all(luma).expect("write Y plane");
    let chroma = vec![128u8; (width / 2) * (height / 2)];
    file.write_all(&chroma).expect("write Cb plane");
    file.write_all(&chroma).expect("write Cr plane");
    file.flush().expect("flush y4m");
    path
}

#[tokio::test]
async fn test_video_reader_decodes_real_frames_and_reports_real_metadata() {
    let path = temp_y4m("decode", 16, 16, 3);

    for backend in enabled_backends() {
        let config = VideoConfig::from_file(path.clone())
            .with_pixel_format(PixelFormat::Gray)
            .with_backend(backend);

        let mut reader = VideoReader::new(config).await.expect("open y4m");

        // Metadata comes from the container, not from hardcoded 30fps /
        // 1920x1080 placeholders.
        let metadata = reader.metadata();
        assert_eq!(metadata.width, 16, "{backend:?} metadata: {metadata:?}");
        assert_eq!(metadata.height, 16, "{backend:?} metadata: {metadata:?}");
        assert!(
            (metadata.fps - 25.0).abs() < 0.01,
            "{backend:?}: expected the container's 25 fps, got {}",
            metadata.fps
        );

        let mut decoded = Vec::new();
        while let Some(frame) = reader.read_frame().await.expect("read_frame") {
            assert_eq!(frame.width, 16, "{backend:?}");
            assert_eq!(frame.height, 16, "{backend:?}");
            assert_eq!(frame.channels, 1, "{backend:?}");
            assert_eq!(frame.data.len(), 16 * 16, "{backend:?}");
            decoded.push(frame);
        }

        assert_eq!(
            decoded.len(),
            3,
            "{backend:?}: read_frame() used to return Ok(None) immediately for every source"
        );
        // The first sample of frame f is ((0 + f*7) % 200) by construction.
        for (f, frame) in decoded.iter().enumerate() {
            let expected = ((f * 7) % 200) as i32;
            assert!(
                (i32::from(frame.data[0]) - expected).abs() <= 2,
                "{backend:?} frame {f}: decoded {} vs written {expected}",
                frame.data[0]
            );
        }
        // Timestamps come from the container's PTS, so they advance; a
        // constant 0.0 would mean `current_time()` is fabricated.
        assert!(
            decoded[1].timestamp > decoded[0].timestamp,
            "{backend:?}: frame timestamps must advance: {:?}",
            decoded.iter().map(|f| f.timestamp).collect::<Vec<_>>()
        );
        assert_eq!(reader.current_frame(), 3, "{backend:?}");
        assert!(
            (reader.current_time() - decoded[2].timestamp).abs() < 1e-9,
            "{backend:?}"
        );
    }

    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn test_video_reader_honours_decimation_and_max_frames() {
    let path = temp_y4m("decimate", 8, 8, 6);

    for backend in enabled_backends() {
        let mut reader = VideoReader::new(
            VideoConfig::from_file(path.clone())
                .with_pixel_format(PixelFormat::Gray)
                .with_decimation(2)
                .with_backend(backend),
        )
        .await
        .expect("open y4m");

        let mut indices = Vec::new();
        while let Some(frame) = reader.read_frame().await.expect("read_frame") {
            indices.push(frame.index);
        }
        assert_eq!(
            indices,
            vec![0, 2, 4],
            "{backend:?}: decimation was never applied before"
        );

        let mut limited = VideoReader::new(
            VideoConfig::from_file(path.clone())
                .with_pixel_format(PixelFormat::Gray)
                .with_max_frames(2)
                .with_backend(backend),
        )
        .await
        .expect("open y4m");
        let mut count = 0;
        while limited.read_frame().await.expect("read_frame").is_some() {
            count += 1;
        }
        assert_eq!(count, 2, "{backend:?}");

        // A one-frame buffer must not truncate the stream: the source is
        // drained repeatedly, including after the demuxer hits EOF.
        let mut tiny = VideoReader::new(
            VideoConfig::from_file(path.clone())
                .with_pixel_format(PixelFormat::Gray)
                .with_buffer_size(1)
                .with_backend(backend),
        )
        .await
        .expect("open y4m");
        let mut tiny_count = 0;
        while tiny.read_frame().await.expect("read_frame").is_some() {
            tiny_count += 1;
        }
        assert_eq!(
            tiny_count, 6,
            "{backend:?}: a small frame buffer must not drop frames"
        );
    }

    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn test_video_reader_resizes_to_target_dimensions() {
    let path = temp_y4m("resize", 16, 16, 1);

    for backend in enabled_backends() {
        let mut reader = VideoReader::new(
            VideoConfig::from_file(path.clone())
                .with_pixel_format(PixelFormat::Rgb)
                .with_resize(8, 4)
                .with_backend(backend),
        )
        .await
        .expect("open y4m");

        let frame = reader
            .read_frame()
            .await
            .expect("read_frame")
            .expect("one frame");
        assert_eq!(
            (frame.width, frame.height, frame.channels),
            (8, 4, 3),
            "{backend:?}"
        );
        assert_eq!(frame.data.len(), 8 * 4 * 3, "{backend:?}");
        assert!(frame.to_array().is_ok(), "{backend:?}");
    }

    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn test_video_reader_rejects_missing_file_instead_of_succeeding() {
    let missing = unique_temp_path("missing", "mp4");
    assert!(!missing.exists());

    for backend in enabled_backends() {
        // Previously `new()` returned Ok for any path whatsoever.
        assert!(
            VideoReader::new(VideoConfig::from_file(missing.clone()).with_backend(backend))
                .await
                .is_err(),
            "{backend:?}: a missing file must not open"
        );
    }
}

// ================================================================
// Backend routing (id=B2 VideoBackend scaffolding): these exercise
// `VideoReader::new`'s backend-selection layer itself, independent of any
// particular backend's decode path.
// ================================================================

/// Forcing `VideoBackend::Pure` at something the pure backend genuinely
/// cannot decode must fail with an honest, specific error -- not a generic
/// "unsupported", not a silent fallback to FFmpeg, and certainly not an `Ok`
/// that decoded nothing. A non-Y4M file is the case that still applies now
/// that Y4M itself works.
#[cfg(feature = "video-pure")]
#[tokio::test]
async fn test_forcing_pure_backend_on_non_y4m_errs_honestly() {
    let path = unique_temp_path("force_pure_non_y4m", "mp4");
    std::fs::write(&path, b"\x00\x00\x00\x20ftypisom-not-really-an-mp4").expect("write fake mp4");

    let result =
        VideoReader::new(VideoConfig::from_file(path.clone()).with_backend(VideoBackend::Pure))
            .await;

    let _ = std::fs::remove_file(&path);

    // `VideoReader` intentionally does not derive `Debug` (it would force
    // the FFmpeg FFI types inside `FfmpegReader` to too), so this checks
    // the `Result` by hand instead of `expect_err`/`unwrap_err`.
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("forcing VideoBackend::Pure at a non-Y4M file must fail"),
    };
    assert!(
        matches!(err, IoError::Unsupported(_)),
        "expected IoError::Unsupported, got: {err:?}"
    );
    let message = err.to_string();
    assert!(
        message.contains("Y4M"),
        "the error must name what the pure backend does support, got: {message}"
    );
}

/// The positive counterpart: forcing `VideoBackend::Pure` at a Y4M file
/// really does decode it through the pure-Rust stack -- including in a build
/// that also has `video`, where nothing may quietly hand the work to FFmpeg.
#[cfg(feature = "video-pure")]
#[tokio::test]
async fn test_forcing_pure_backend_decodes_y4m() {
    let path = temp_y4m("force_pure_y4m", 4, 4, 2);

    let mut reader = VideoReader::new(
        VideoConfig::from_file(path.clone())
            .with_pixel_format(PixelFormat::Gray)
            .with_backend(VideoBackend::Pure),
    )
    .await
    .expect("the pure backend must open a Y4M file");

    let mut frames = Vec::new();
    while let Some(frame) = reader.read_frame().await.expect("read_frame") {
        frames.push(frame);
    }

    let _ = std::fs::remove_file(&path);

    assert_eq!(frames.len(), 2);
    assert_eq!(
        (frames[0].width, frames[0].height, frames[0].channels),
        (4, 4, 1)
    );
    assert_eq!(frames[0].data[0], 0, "Cmono luma must be copied verbatim");
}

/// The three-output-format conversion matrix, exercised end to end through
/// `VideoReader` against a real 4:2:0 file -- the chroma-bearing path that
/// the `Cmono` parity fixture above never reaches.
///
/// Pure-backend-only: libswscale and `oximedia-core` round YUV -> RGB
/// slightly differently, so exact colour equality is not part of the
/// cross-backend contract. The invariants asserted here (luma passed
/// through untouched for `Gray`, achromatic output for neutral chroma,
/// opaque alpha for `Rgba`) hold for any correct converter.
#[cfg(feature = "video-pure")]
#[tokio::test]
async fn test_pure_backend_converts_420_to_every_pixel_format() {
    let luma: Vec<u8> = (0..16u32).map(|i| (i * 15 + 16) as u8).collect();
    let path = temp_y4m_420("pure_420", 4, 4, &luma);

    let open = |format| {
        VideoReader::new(
            VideoConfig::from_file(path.clone())
                .with_pixel_format(format)
                .with_backend(VideoBackend::Pure),
        )
    };

    let mut gray = open(PixelFormat::Gray).await.expect("open 420 as Gray");
    let gray_frame = gray
        .read_frame()
        .await
        .expect("read_frame")
        .expect("one frame");
    assert_eq!(gray_frame.channels, 1);
    assert_eq!(
        gray_frame.data, luma,
        "Gray must be the Y plane verbatim -- no BT.601 range expansion"
    );

    let mut rgb = open(PixelFormat::Rgb).await.expect("open 420 as Rgb");
    let rgb_frame = rgb
        .read_frame()
        .await
        .expect("read_frame")
        .expect("one frame");
    assert_eq!(rgb_frame.channels, 3);
    assert_eq!(rgb_frame.data.len(), 4 * 4 * 3);
    for (pixel, y) in rgb_frame.data.chunks_exact(3).zip(luma.iter()) {
        assert!(
            pixel[0] == pixel[1] && pixel[1] == pixel[2],
            "neutral chroma must decode achromatic, got {pixel:?} for luma {y}"
        );
    }
    // Luma is limited-range here, so RGB *is* expanded (unlike Gray): the
    // 16..=235 studio range maps onto 0..=255.
    assert_eq!(rgb_frame.data[0], 0, "Y=16 is full-range black");

    let mut rgba = open(PixelFormat::Rgba).await.expect("open 420 as Rgba");
    let rgba_frame = rgba
        .read_frame()
        .await
        .expect("read_frame")
        .expect("one frame");
    assert_eq!(rgba_frame.channels, 4);
    assert_eq!(rgba_frame.data.len(), 4 * 4 * 4);
    for (rgba_pixel, rgb_pixel) in rgba_frame
        .data
        .chunks_exact(4)
        .zip(rgb_frame.data.chunks_exact(3))
    {
        assert_eq!(&rgba_pixel[..3], rgb_pixel, "RGBA must widen RGB unchanged");
        assert_eq!(
            rgba_pixel[3], 255,
            "Y4M carries no alpha; 255 is the only honest value"
        );
    }

    let _ = std::fs::remove_file(&path);
}

/// `start_time` and [`VideoReader::seek`] really reposition the stream --
/// they are the only two pure-backend paths that consume frames without
/// delivering them, so nothing else would catch an off-by-one there.
///
/// Pure-backend-only on purpose: FFmpeg seeks to a keyframe boundary in
/// container time, so landing on the exact frame index is not part of the
/// cross-backend contract the four parity tests above enforce.
#[cfg(feature = "video-pure")]
#[tokio::test]
async fn test_pure_backend_start_time_and_seek_reposition() {
    // Written at 25 fps, so raw frame n sits at n * 0.04 s.
    let path = temp_y4m("pure_seek", 4, 4, 6);

    let mut reader = VideoReader::new(
        VideoConfig::from_file(path.clone())
            .with_pixel_format(PixelFormat::Gray)
            .with_backend(VideoBackend::Pure)
            .with_start_time(0.08),
    )
    .await
    .expect("open y4m with start_time");

    let frame = reader
        .read_frame()
        .await
        .expect("read_frame")
        .expect("a frame at 0.08s");
    assert_eq!(frame.index, 2, "start_time must skip the first two frames");
    assert!((frame.timestamp - 0.08).abs() < 1e-9);

    // Rewind to the top: the file is reopened, so frame 0 comes back.
    reader.seek(0.0).await.expect("seek to 0");
    let first = reader
        .read_frame()
        .await
        .expect("read_frame")
        .expect("frame 0");
    assert_eq!(first.index, 0);
    assert!((first.timestamp - 0.0).abs() < 1e-9);

    // Forward seek lands on the frame boundary and reports it.
    reader.seek(0.16).await.expect("seek forward");
    assert!((reader.current_time() - 0.16).abs() < 1e-9);
    let fifth = reader
        .read_frame()
        .await
        .expect("read_frame")
        .expect("frame 4");
    assert_eq!(fifth.index, 4);

    // Nonsensical targets are a config error, never a panic.
    assert!(reader.seek(-1.0).await.is_err());
    assert!(reader.seek(f64::NAN).await.is_err());
    assert!(reader.seek(f64::INFINITY).await.is_err());

    let _ = std::fs::remove_file(&path);
}

/// The capability probe `Auto` resolution consults: `true` for a Y4M file,
/// `true` for a camera where `oximedia-capture` has a backend, `false` for
/// the sources the pure backend still cannot open.
///
/// This is deliberately *not* a routing-introspection test. Which backend
/// `Auto` picked is not observable from outside -- `ReaderImpl` is private
/// and `VideoReader` exposes no backend accessor. The probe is asserted
/// directly here; the routing rule it feeds is documented in the `video`
/// module's "Backend selection" section, and the observable *consequence*
/// of that rule (an `Auto`-opened Y4M file behaving exactly like a
/// forced-`Pure` one) is asserted separately by
/// [`test_auto_routes_y4m_to_pure_observably`] below.
#[cfg(feature = "video-pure")]
#[test]
fn test_pure_backend_capability_probe() {
    let path = temp_y4m("auto_route", 4, 4, 1);
    let config = VideoConfig::from_file(path.clone());
    assert_eq!(config.backend, VideoBackend::Auto);

    assert!(
        backend_pure::PureReader::supports(&config.source),
        "the pure backend must report itself capable of a Y4M file"
    );
    // The camera answer tracks whether `oximedia-capture` has a backend for
    // this target: `true` where it does (Linux/macOS/Windows, via
    // V4L2/AVFoundation/Media Foundation respectively), `false` where it
    // does not -- because claiming the source there
    // would take cameras away from FFmpeg only to fail at open time. Asserted
    // against `backend_kind()` rather than hardcoded per OS so this test says
    // the same thing on every host and stays correct as backends land.
    assert_eq!(
        backend_pure::PureReader::supports(&VideoSource::Camera("0".to_string())),
        oximedia_capture::backend_kind().is_available(),
        "the pure backend must claim a camera exactly where it has a capture backend"
    );
    assert!(
        !backend_pure::PureReader::supports(&VideoSource::Network(
            "rtsp://example.invalid/s".to_string()
        )),
        "network streams are not implemented in the pure backend"
    );

    let _ = std::fs::remove_file(&path);
}

/// The observable half of `Auto` routing: opening the same Y4M file once
/// through the default `Auto` backend and once forced onto
/// [`VideoBackend::Pure`] must decode to the same frame sequence.
///
/// `VideoReader` has no backend accessor (see `ReaderImpl`'s doc comment),
/// so this cannot assert "`Auto` picked `Pure`" directly -- what it asserts
/// is the observable consequence that picking anything else would break.
/// The fixture is 4:2:0 (`temp_y4m_420`) converted to RGB, rather than the
/// `Cmono`/Gray parity fixture the four tests above use: those are exact
/// byte-for-byte contract, but the RGB path is not, and this test needs
/// that slack to have any discriminating power at all. Empirically (checked
/// by hand while writing this test, against this exact fixture, under
/// `--features video,video-pure`): forced-`Pure` and forced-`Ffmpeg` decode
/// it to *different* RGB bytes, even though `temp_y4m_420` writes neutral
/// (128) chroma -- the divergence survives zeroing out the chroma terms
/// because the two backends' fixed-point luma scaling
/// (`oximedia-core`'s `convert::pixel::yuv420_to_rgb`, ×1024 fixed point,
/// vs libswscale's) already rounds slightly differently. If `Auto` ever
/// silently fell back to FFmpeg for a Y4M file with `video-pure` compiled
/// in, this test would fail on the byte comparison below even though it
/// never inspects which backend actually ran. This is an empirical property
/// of the current implementations, not a documented contract -- if a future
/// change makes the two converters agree on this fixture, the test stays
/// correct (still passes) but loses this discriminating power silently;
/// nothing here re-verifies the divergence still holds.
///
/// This does not prove the two converters always agree on every input --
/// they are explicitly not held to pixel-exact parity -- only that, for
/// this fixture, `Auto` and forced-`Pure` produce identical output, which
/// is the strongest evidence obtainable without a backend accessor. In a
/// `video-pure`-only build (no FFmpeg compiled in at all) this degenerates
/// to "`Auto` opens and decodes a Y4M file at all", which is still worth
/// asserting, just without the discriminating power the dual-feature build
/// gets from a second backend actually being available to fall back to.
#[cfg(feature = "video-pure")]
#[tokio::test]
async fn test_auto_routes_y4m_to_pure_observably() {
    async fn read_all(path: &std::path::Path, backend: VideoBackend) -> Vec<VideoFrame> {
        let mut reader = VideoReader::new(
            VideoConfig::from_file(path)
                .with_pixel_format(PixelFormat::Rgb)
                .with_backend(backend),
        )
        .await
        .unwrap_or_else(|error| panic!("{backend:?}: open must succeed: {error}"));

        let mut frames = Vec::new();
        while let Some(frame) = reader.read_frame().await.expect("read_frame") {
            frames.push(frame);
        }
        frames
    }

    let luma: Vec<u8> = (0..16u32).map(|i| (i * 15 + 16) as u8).collect();
    let path = temp_y4m_420("auto_vs_pure", 4, 4, &luma);

    let auto_frames = read_all(&path, VideoBackend::Auto).await;
    let pure_frames = read_all(&path, VideoBackend::Pure).await;

    let _ = std::fs::remove_file(&path);

    assert_eq!(
        auto_frames.len(),
        pure_frames.len(),
        "Auto and forced-Pure must decode the same number of frames"
    );
    // `VideoFrame` derives `Clone, Debug` but not `PartialEq`, so frames are
    // compared field by field rather than with `assert_eq!` on the whole
    // struct.
    for (index, (auto_frame, pure_frame)) in auto_frames.iter().zip(pure_frames.iter()).enumerate()
    {
        assert_eq!(auto_frame.index, pure_frame.index, "frame {index}");
        assert_eq!(auto_frame.width, pure_frame.width, "frame {index}");
        assert_eq!(auto_frame.height, pure_frame.height, "frame {index}");
        assert_eq!(auto_frame.channels, pure_frame.channels, "frame {index}");
        assert!(
            (auto_frame.timestamp - pure_frame.timestamp).abs() < 1e-9,
            "frame {index}: {} vs {}",
            auto_frame.timestamp,
            pure_frame.timestamp
        );
        assert_eq!(
            auto_frame.data, pure_frame.data,
            "frame {index}: Auto and forced-Pure must decode identical bytes"
        );
    }
}

// ================================================================
// Camera capture (id=B4): the pure backend's live-source path, driven by
// `oximedia-capture`'s deterministic `mock` backend so that every assertion
// below runs on a machine with no camera attached (and without ever putting
// a TCC permission dialog on screen).
//
// The mock is opened directly and handed to `PureReader::from_capture_session`
// -- `VideoReader::new` deliberately cannot be pointed at it, because a
// production path that could be redirected to a synthetic device would be a
// production path that can silently deliver synthetic frames. Everything
// after that seam is the code a real device runs.
// ================================================================

/// Open a scripted mock capture session.
#[cfg(feature = "video-pure")]
fn mock_session(frames: usize, width: u32, height: u32) -> oximedia_capture::CaptureSession {
    use oximedia_capture::{mock, CaptureConfig};

    mock::open(
        CaptureConfig::default().with_queue_depth(8),
        mock::MockScript::default()
            .with_frames(frames)
            .with_size(width, height)
            // Flat out: nothing here is timing-dependent, and a cadence would
            // only make the suite slower.
            .with_cadence(std::time::Duration::ZERO),
    )
    .expect("the mock device always exists")
}

/// Build a `PureReader` over a scripted mock camera.
#[cfg(feature = "video-pure")]
fn mock_camera_reader(
    config: VideoConfig,
    frames: usize,
    width: u32,
    height: u32,
) -> backend_pure::PureReader {
    match backend_pure::PureReader::from_capture_session(
        config,
        mock_session(frames, width, height),
    ) {
        Ok(reader) => reader,
        Err(error) => panic!("the mock capture session must open: {error}"),
    }
}

/// End to end over a live-shaped source: every scripted frame arrives, with
/// the negotiated geometry, the configured channel count and a monotonic
/// timeline, and the stream then ends cleanly.
#[cfg(feature = "video-pure")]
#[tokio::test]
async fn test_pure_backend_camera_delivers_every_scripted_frame() {
    let mut reader = mock_camera_reader(VideoConfig::from_camera("mock:0"), 5, 64, 48);

    // Metadata comes from the negotiated format, not from the request.
    let metadata = reader.metadata();
    assert_eq!((metadata.width, metadata.height), (64, 48));
    assert!((metadata.fps - 30.0).abs() < 1e-9, "{}", metadata.fps);
    assert_eq!(
        (metadata.duration, metadata.frame_count),
        (0.0, 0),
        "a live device has neither a duration nor a frame count"
    );

    let mut frames = Vec::new();
    while let Some(frame) = reader.read_frame().await.expect("read_frame") {
        assert_eq!((frame.width, frame.height), (64, 48));
        assert_eq!(
            frame.channels, 3,
            "PixelFormat::Rgb is the from_camera default"
        );
        assert_eq!(frame.data.len(), 64 * 48 * 3);
        assert!(frame.to_array().is_ok());
        frames.push(frame);
    }

    assert_eq!(frames.len(), 5, "every scripted frame must be delivered");
    let indices: Vec<u64> = frames.iter().map(|frame| frame.index).collect();
    assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    // The mock's device clock runs at the nominal frame period, and the
    // session normalises it to start at zero.
    assert!((frames[0].timestamp - 0.0).abs() < 1e-9);
    for pair in frames.windows(2) {
        assert!(
            pair[1].timestamp > pair[0].timestamp,
            "capture timestamps must advance: {:?}",
            frames.iter().map(|f| f.timestamp).collect::<Vec<_>>()
        );
    }
    assert_eq!(reader.current_frame(), 5);
    assert!((reader.current_time() - frames[4].timestamp).abs() < 1e-9);

    // Past the end of the script the stream is finished, and stays finished.
    assert!(reader.read_frame().await.expect("read_frame").is_none());
}

/// Decimation on a live source counts *received* frames, exactly as it counts
/// demuxed ones on a file: raw indices 0, 2, 4 survive a decimation of 2.
#[cfg(feature = "video-pure")]
#[tokio::test]
async fn test_pure_backend_camera_honours_decimation() {
    let mut reader = mock_camera_reader(
        VideoConfig::from_camera("mock:0").with_decimation(2),
        6,
        16,
        16,
    );

    let mut indices = Vec::new();
    while let Some(frame) = reader.read_frame().await.expect("read_frame") {
        indices.push(frame.index);
    }
    assert_eq!(indices, vec![0, 2, 4]);
}

/// `max_frames` stops delivery on a camera the same way it does on a file:
/// it counts frames handed to the caller, not frames received.
#[cfg(feature = "video-pure")]
#[tokio::test]
async fn test_pure_backend_camera_honours_max_frames() {
    let mut reader = mock_camera_reader(
        VideoConfig::from_camera("mock:0").with_max_frames(2),
        6,
        16,
        16,
    );

    let mut delivered = 0;
    while reader.read_frame().await.expect("read_frame").is_some() {
        delivered += 1;
    }
    assert_eq!(delivered, 2);
    assert_eq!(reader.current_frame(), 2);
}

/// A one-frame buffer must not truncate a live stream -- the camera pump
/// takes one frame per call precisely so a small buffer costs nothing.
#[cfg(feature = "video-pure")]
#[tokio::test]
async fn test_pure_backend_camera_survives_a_tiny_buffer() {
    let mut reader = mock_camera_reader(
        VideoConfig::from_camera("mock:0").with_buffer_size(1),
        4,
        16,
        16,
    );

    let mut delivered = 0;
    while reader.read_frame().await.expect("read_frame").is_some() {
        delivered += 1;
    }
    assert_eq!(delivered, 4);
}

/// A script that emits nothing ends immediately, rather than blocking or
/// inventing a frame.
#[cfg(feature = "video-pure")]
#[tokio::test]
async fn test_pure_backend_camera_ends_cleanly_on_an_empty_script() {
    let mut reader = mock_camera_reader(VideoConfig::from_camera("mock:0"), 0, 16, 16);
    assert!(reader.read_frame().await.expect("read_frame").is_none());
    assert_eq!(reader.current_frame(), 0);
}

/// Conversion and rescaling apply to captured frames too: the mock's Gray8
/// frames come out in whichever `PixelFormat` was configured, at the
/// requested size.
#[cfg(feature = "video-pure")]
#[tokio::test]
async fn test_pure_backend_camera_converts_and_rescales() {
    for (format, channels) in [
        (PixelFormat::Gray, 1usize),
        (PixelFormat::Rgb, 3),
        (PixelFormat::Rgba, 4),
    ] {
        let mut reader = mock_camera_reader(
            VideoConfig::from_camera("mock:0")
                .with_pixel_format(format)
                .with_resize(8, 4),
            1,
            32,
            16,
        );

        let frame = reader
            .read_frame()
            .await
            .expect("read_frame")
            .expect("one frame");
        assert_eq!(
            (frame.width, frame.height, frame.channels),
            (8, 4, channels)
        );
        assert_eq!(frame.data.len(), 8 * 4 * channels);
        if matches!(format, PixelFormat::Rgba) {
            for pixel in frame.data.chunks_exact(4) {
                assert_eq!(pixel[3], 255, "captured frames carry no alpha");
            }
        }
        // The mock paints a moving gradient, so a frame of a single value
        // would mean the payload never reached the conversion.
        assert!(
            frame.data.iter().any(|&byte| byte != frame.data[0]),
            "{format:?}: the captured gradient must survive conversion"
        );
        assert_eq!(reader.metadata().width, 8);
    }
}

/// A live capture device cannot be seeked, and says so rather than silently
/// discarding frames until the timestamp passes.
#[cfg(feature = "video-pure")]
#[tokio::test]
async fn test_pure_backend_camera_refuses_to_seek() {
    let mut reader = mock_camera_reader(VideoConfig::from_camera("mock:0"), 2, 16, 16);

    for target in [0.0, 1.5, -1.0, f64::NAN] {
        let error = match reader.seek(target).await {
            Err(error) => error,
            Ok(()) => panic!("seeking a camera to {target} must fail"),
        };
        assert!(matches!(error, IoError::Unsupported(_)), "{error:?}");
        assert!(error.to_string().contains("live capture device"), "{error}");
    }
}

/// `start_time` is the same refusal at open time: there is no past to start
/// from, and skipping frames off the front of a live stream would discard
/// them silently.
#[cfg(feature = "video-pure")]
#[test]
fn test_pure_backend_camera_refuses_start_time() {
    let config = VideoConfig::from_camera("mock:0").with_start_time(1.0);
    let error =
        match backend_pure::PureReader::from_capture_session(config, mock_session(2, 16, 16)) {
            Err(error) => error,
            Ok(_) => panic!("start_time on a camera must fail"),
        };
    assert!(matches!(error, IoError::Unsupported(_)), "{error:?}");
    assert!(error.to_string().contains("start_time"), "{error}");
}

/// A scripted fatal backend failure surfaces as an error rather than a silent
/// end-of-stream, and the session is finished afterwards.
#[cfg(feature = "video-pure")]
#[tokio::test]
async fn test_pure_backend_camera_reports_a_fatal_capture_error() {
    use oximedia_capture::{mock, CaptureConfig};

    let session = mock::open(
        CaptureConfig::default(),
        mock::MockScript::default()
            .with_frames(4)
            .with_cadence(std::time::Duration::ZERO)
            .with_fail_at(0),
    )
    .expect("the mock device always exists");

    let mut reader = match backend_pure::PureReader::from_capture_session(
        VideoConfig::from_camera("mock:0"),
        session,
    ) {
        Ok(reader) => reader,
        Err(error) => panic!("the mock capture session must open: {error}"),
    };

    let error = match reader.read_frame().await {
        Err(error) => error,
        Ok(frame) => panic!(
            "a scripted capture failure must not read as {:?}",
            frame.is_some()
        ),
    };
    // `CaptureError::Platform` -> `IoError::ReadFailed`, per the mapping
    // table in `backend_pure_camera::map_capture_error`.
    assert!(matches!(error, IoError::ReadFailed(_)), "{error:?}");
    assert!(error.to_string().contains("scripted failure"), "{error}");

    // The stream reports its end once the error has been delivered.
    assert!(reader.read_frame().await.expect("read_frame").is_none());
}

/// `Auto` must report the same "file not found" error regardless of which
/// backend feature(s) are compiled in -- the probe that detects a missing
/// file runs before any backend is chosen.
#[cfg(feature = "video-pure")]
#[tokio::test]
async fn test_auto_rejects_missing_file_with_only_pure_backend_compiled() {
    let missing = unique_temp_path("pure_missing", "y4m");
    assert!(!missing.exists());
    assert!(VideoReader::new(VideoConfig::from_file(missing))
        .await
        .is_err());
}

/// `VideoBackend` defaults to `Auto`, and a `VideoConfig` serialized before
/// this field existed (i.e. JSON missing the `backend` key) must still
/// deserialize to `Auto` -- that is what `#[serde(default)]` buys.
#[test]
fn test_video_backend_default_is_auto_and_missing_field_deserializes() {
    assert_eq!(VideoBackend::default(), VideoBackend::Auto);

    let config = VideoConfig::from_file("video.mp4");
    assert_eq!(config.backend, VideoBackend::Auto);

    let mut value = serde_json::to_value(&config).expect("VideoConfig must serialize");
    value
        .as_object_mut()
        .expect("VideoConfig serializes to a JSON object")
        .remove("backend");
    let json_without_backend = serde_json::to_string(&value).expect("re-encode as JSON text");

    let round_tripped: VideoConfig = serde_json::from_str(&json_without_backend)
        .expect("VideoConfig without `backend` must still deserialize (#[serde(default)])");
    assert_eq!(round_tripped.backend, VideoBackend::Auto);
}
