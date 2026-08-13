# kizzasi-io

![status](https://img.shields.io/badge/status-stable-brightgreen)

Physical world connectors for Kizzasi - MQTT, Audio, and sensor streams.

## Overview

Comprehensive I/O toolkit for real-time signal acquisition and processing. Connects Kizzasi to sensors, audio devices, network protocols, and file formats.

## Features

- **MQTT** (`mqtt`, default-on): QoS, wildcard topics, reconnection logic; TLS behind the opt-in `mqtt-tls` feature
- **Audio** (`audio`, opt-in): CPAL, JACK, ASIO backends with multi-channel support
- **Video** (`video`): FFmpeg demux/decode of files and network streams (any container the linked FFmpeg supports), rescaling and pixel-format conversion, decimation, seeking, optical flow. Camera capture goes through `libavdevice` (v4l2 / DirectShow / AVFoundation). A pure-Rust alternative, **`video-pure`** (the OxiMedia stack, no C linked), decodes Y4M files end to end and captures from cameras on Linux/macOS/Windows; `VideoBackend::Auto` (the default) prefers it for Y4M files and cameras whenever it is compiled in. Camera *enumeration* is real on every platform when `video-pure` is compiled in (preferred even alongside `video`, since the FFmpeg path cannot match it); in a `video`-only build it stays Linux-only and returns `IoError::Unsupported` elsewhere rather than inventing a device list
- **Signal Processing** (always available): FFT, filters (IIR/FIR), wavelets, MFCC extraction
- **Network**: WebSocket (`websocket`, with ping/pong keepalive), TCP/UDP (`network`), serial (`serial`), OSC (`osc`), ZeroMQ (`zeromq`, pure-Rust ZMTP: PUB/SUB with topic filtering, PUSH/PULL, REQ/REP, DEALER)
- **File I/O** (`file`): WAV and CSV; HDF5 behind the opt-in `hdf5` feature (pure Rust via OxiH5)
- **Recording & Playback** (always available): `StreamRecorder`/`StreamPlayer` with Binary, JSON, and CSV formats
- **Stream Sync** (always available): Multi-stream time alignment (`StreamSynchronizer`), PLL clock recovery, NTP/PTP-style `TimeSynchronizer`
- **Advanced DSP** (always available): Hilbert-Huang Transform, beamforming, source separation
- **Performance** (always available, `simd` for vectorized paths): Lock-free queues, zero-copy buffers, SIMD operations (`rms_simd`/`normalize_simd`/`add_simd`/`multiply_simd` dispatch to `scirs2-core`'s vectorized kernels over borrowed views)

### Cargo Feature Flags

Only `std` and `mqtt` are enabled by default (`default = ["std", "mqtt"]`).
Everything else in the table below is opt-in: add the feature name(s) you
need to your `Cargo.toml`, e.g.
`kizzasi-io = { version = "0.2", features = ["serial", "osc"] }`. Without
the matching feature, the corresponding module and its re-exported types
(`SerialStream`, `WavReader`, `TcpClientStream`, `OscSender`, ...) do not
exist in the crate at all.

The **Pure Rust** column says whether the feature keeps the build free of
C/C++/Fortran. Everything on by default is pure Rust; the three impure
features are opt-in precisely so that the default build links no native
library. Enabling one is a deliberate trade, not an accident.

| Feature | Default | Pure Rust | Enables | Key types |
|---|---|---|---|---|
| `std` | On | Yes | Standard-library support | — |
| `mqtt` | On | Yes | MQTT client over plain TCP (`src/mqtt.rs`) | `MqttClient`, `MqttConfig`, `MqttStream` |
| `mqtt-tls` | Off | Yes — rustls + the pure-Rust RustCrypto provider (`oxitls-rustcrypto-provider`), injected explicitly instead of resolving rustls's aws-lc-rs default (no C/assembly compiled). On Apple/Windows this still links the OS certificate-store framework (Security.framework / schannel) via `rustls-native-certs`, a link kizzasi itself never triggers | TLS transport for the MQTT client | `TlsConfig` becomes effective |
| `websocket` | Off | Yes | WebSocket client (`src/websocket.rs`) | `WebSocketStream`, `WebSocketConfig` |
| `serial` | Off | Yes | Serial port I/O (`src/serial.rs`) | `SerialStream`, `SerialConfig` |
| `file` | Off | Yes | WAV/CSV file I/O (`src/file.rs`) | `WavReader`, `WavWriter`, `CsvReader`, `CsvWriter` |
| `hdf5` | Off | Yes — pure Rust via OxiH5 (no libhdf5 install) | HDF5 dataset I/O (`src/file.rs`); implies `file` | `Hdf5Reader`, `Hdf5Writer` |
| `network` | Off | Yes | Raw TCP/UDP sockets (`src/socket.rs`) | `TcpClientStream`, `TcpServerStream`, `UdpSocketStream` |
| `osc` | Off | Yes | Open Sound Control (`src/osc.rs`) | `OscSender`, `OscReceiver`, `OscServer` |
| `zeromq` | Off | Yes | ZeroMQ messaging over the pure-Rust `zeromq` (zmq.rs) ZMTP stack -- no `libzmq` (`src/zeromq.rs`) | `ZmqStream`, `ZmqConfig` |
| `ros2` | Off | Yes | ROS2 bridge over the pure-Rust `ros2-client`/RustDDS stack -- no ROS2 installation required, available on every platform (`src/ros2.rs`) | `Ros2Stream`, `Ros2Config` |
| `simd` | Off | Yes | SIMD-accelerated signal processing paths | — |
| `metal` | Off | Apple system FFI | candle Metal backend (propagated to `kizzasi-core`); Apple platforms only | — |
| `audio` | Off | **No** — cpal links libasound (`alsa-sys`) on Linux, CoreAudio on Apple, WASAPI on Windows | Live audio device I/O (`src/audio.rs`) | `AudioInput`, `AudioOutput`, `AudioConfig` |
| `video` | Off | **No** — binds the FFmpeg C libraries | FFmpeg-backed video decoding and capture (`src/video/`); see `video-pure` for the pure-Rust backend | `VideoReader`, `VideoBackend`, `VideoProcessor`, `CameraDevice` |
| `video-pure` | Off | Yes — `oximedia-container`/`oximedia-core`/`oximedia-cv`/`oximedia-capture`/`oximedia-codec`/`oximedia-simd`, no C compiled or linked | Pure-Rust video backend (`src/video/`, OxiMedia stack). **Y4M (YUV4MPEG2) files**: decoded end to end -- demux, YUV→RGB/RGBA/Gray conversion and bilinear rescaling -- matching the FFmpeg backend's *observable* contract for that container (frame indices, timestamps, decimation/buffering/`max_frames` accounting); the two backends' YUV→RGB conversions are not guaranteed bit-identical for chroma-bearing content. **Camera capture**: works on Linux (V4L2), macOS (AVFoundation) and Windows (Media Foundation) -- every platform `oximedia-capture` implements a backend for. `oximedia-capture` negotiates a mode, preferring NV12 first (cheapest to convert), then the packed 4:2:2 layouts (YUYV/UYVY), then planar 4:2:0 and packed RGB24, with MJPEG last (a full JPEG decode per frame); `oximedia-codec` decodes MJPEG, `oximedia-simd` converts the packed 4:2:2 rows. `seek`/`start_time` are refused on a live device rather than silently discarding frames. Network streams and other containers return an honest `Err`. Also replaces `CameraDevice::list_devices` with real enumeration on **every** platform, even when `video` is enabled -- on Linux this is capability-filtered (`VIDIOC_QUERYCAP`) rather than the `video`-only path's unconditional `/dev/video*` scan. With `video` also enabled, `Auto` sends Y4M files and cameras here and everything else to FFmpeg | `VideoReader`, `VideoBackend`, `CameraDevice` |

There is no `cuda` feature: candle's CUDA backend needs an NVIDIA toolkit at
build time, which cannot be expressed as a Cargo feature without breaking
every build that lacks one. See `kizzasi-core`'s `Cargo.toml` for the full
reasoning.

`TlsConfig` and `MqttConfig::enable_tls` exist in every build so
configuration files round-trip, but a config with `use_tls = true` in a build
without `mqtt-tls` fails loudly at `connect()` — it never downgrades to a
plaintext connection.

Signal processing (FFT, filters, wavelets, cepstral analysis, adaptive
filters, beamforming, source separation, resampling, quality metrics),
recording/playback, stream synchronization, compression, calibration, and
the lock-free/zero-copy buffer types are all unconditional -- no feature
flag needed.

## Quick Start

```rust
use kizzasi_io::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Audio input at 16kHz
    let audio_config = AudioConfig::new().sample_rate(16000).channels(1);
    let mut audio = AudioInput::new(audio_config)?;
    audio.start()?;

    // Read audio samples (via the SignalStream trait). Streams never
    // fabricate silence: `read()` returns `Err(IoError::BufferEmpty)` until a
    // full block has been captured, and `Err(IoError::EndOfStream)` once the
    // source is exhausted. See "Stream read contract" below.
    let samples = loop {
        match audio.read() {
            Ok(block) => break block,
            Err(IoError::BufferEmpty) => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(e) => return Err(e.into()),
        }
    };

    // Apply filtering
    let mut processor = SignalProcessor::new(samples.len()).with_sample_rate(16000.0);
    let filtered = processor.apply_filter(
        &samples,
        Filter::LowPass {
            cutoff: 3000.0,
            order: 4,
        },
    )?;

    // MQTT streaming
    let mqtt_config = MqttConfig::new("broker.local", 1883).client_id("sensor");
    let mut client = MqttClient::new(mqtt_config, StreamConfig::new());
    client.connect().await?;
    let payload: Vec<u8> = filtered.iter().flat_map(|f| f.to_le_bytes()).collect();
    client
        .publish("readings/temperature", payload, QosLevel::AtLeastOnce, false)
        .await?;

    Ok(())
}
```

## Stream read contract

`SignalStream::read()` / `AsyncSignalStream::read()` never zero-pad and never
fabricate silence. Every implementation in this crate returns exactly one of:

| Outcome | Meaning |
|---|---|
| `Ok(buffer)` | Only samples that genuinely arrived. A live source yields a full `buffer_size` block; a finite source (file, in-memory buffer, or a closed live source) yields its remaining samples as one final short block. |
| `Err(IoError::BufferEmpty)` | A full block is not available yet and more samples may still arrive. Nothing was consumed, so retrying later is lossless. |
| `Err(IoError::EndOfStream)` | The stream is permanently exhausted or closed and fully drained; `is_active()` is `false` from then on. |

`Ros2Stream` additionally returns `Err(IoError::NotConnected(..))` before
`start()` has created the subscription.

## Signal Processing Subsystems

### Basic Filters

- **FIR**: Sinc lowpass/highpass, moving average, differentiator
- **IIR**: Butterworth lowpass/highpass, notch filters

### Cepstral Analysis

- `RealCepstrum` — pitch detection and voice/unvoiced classification via real cepstrum
- `ComplexCepstrum` — homomorphic deconvolution for source/filter separation
- `FormantTracker` — speech resonance (formant) detection and tracking
- `QuefrencyFilter` — liftering in the quefrency domain for smooth spectral envelopes
- `CepstralDistance` — objective speech quality and similarity assessment

### Advanced Time-Frequency Transforms

- `GaborTransform` — Gaussian-windowed STFT for optimal time-frequency resolution
- `STransform` — frequency-dependent resolution (Stockwell transform)
- `WignerVille` — high-resolution quadratic time-frequency distribution
- `ChoiWilliams` — exponential kernel distribution with cross-term suppression
- `ReassignedSpectrogram` — sharpened energy localization via reassignment

### Adaptive Filters

- `KalmanFilter` — optimal linear state estimation with prediction/update cycles
- `ParticleFilter` — non-Gaussian/nonlinear Bayesian estimation via particle sets
- `LmsFilter` — Least Mean Squares adaptive filter
- `NlmsFilter` — Normalized LMS for improved convergence stability
- `RlsFilter` — Recursive Least Squares for rapid tracking adaptation

### Microphone Array Processing

- `MicrophoneArray` — multi-channel array geometry management and calibration
- `DelayAndSum` — classical broadside/steered delay-and-sum beamforming
- `AdaptiveBeamformer` — MVDR/LCMV null-steering adaptive beamformer
- `DOAEstimator` — Direction-of-Arrival estimation via steered response power (SRP) scanning

### Speech Quality Metrics

- `SnrCalculator` — overall, segmental and frequency-weighted signal-to-noise ratio
- `StoiCalculator` — Short-Time Objective Intelligibility, implementing the algorithm of
  Taal et al. (2011): silent-frame removal, STFT, 15 one-third octave bands, 30-frame
  segments, SDR clipping, mean band-envelope correlation
- `PesqCalculator` — PESQ-*inspired* perceptual score on a 1.0–4.5 scale (band loudness
  spectra with masking and asymmetry weighting)
- `PolqaCalculator` — POLQA-*inspired* score on a 1.0–5.0 scale
- `MosPredictor` — comparative quality index blending SNR, the PESQ-like score and STOI

> **Not ITU-conformant.** `PesqCalculator` and `PolqaCalculator` implement this crate's own
> perceptual model, not ITU-T P.862 / P.863: there is no IRS filtering, no utterance-based
> time alignment, no Bark-scale auditory transform and no published regression onto the
> MOS-LQO scale. `StoiCalculator` implements the published STOI algorithm but has not been
> validated against the reference implementation, and it analyses signals at their native
> sample rate instead of resampling to 10 kHz. Each of these types (and `MosPredictor`)
> exposes `STANDARD_CONFORMANT = false`. Use the scores to rank degradations of the same
> reference against each other — never as certified MOS-LQO numbers.

### Source Separation

- `FastICA` — Independent Component Analysis (fast fixed-point algorithm)
- `NMF` — Non-negative Matrix Factorization for spectrogram decomposition
- `PCA` — Principal Component Analysis for dimensionality reduction and whitening

### Empirical Mode Decomposition

- `EmpiricalModeDecomposition` — adaptive decomposition into Intrinsic Mode Functions (IMFs)
- `EnsembleEmd` — Ensemble EMD (EEMD) for noise-assisted mode extraction

### Advanced Resampling

- `FarrowResampler` — Farrow polynomial structure for fractional-delay resampling
- `ArbitrarySrcResampler` — arbitrary rational sample-rate conversion
- `SincStreamingResampler` — band-limited sinc interpolation in a streaming context
- `TimeVaryingResampler` — instantaneous-rate resampling for pitch-shifting and time-stretching

### Stream Synchronization

- `StreamSynchronizer` — multi-stream time-alignment with configurable tolerance
- `PhaseLockLoop` — PLL-based clock recovery and synchronization
- `TimeSynchronizer` — NTP/PTP-style timestamp reconciliation across streams

### Stream Multiplexing / Demultiplexing

- `StreamMultiplexer` — round-robin, time-ordered, and weighted merging of input streams
- `StreamDemultiplexer` — channel-splitting and routing of multiplexed streams

### Signal Calibration

- `CalibrationManager` — unified calibration session management and persistence
- `MultiPointCalibrator` — piecewise linear / polynomial multi-point calibration
- `AutoCalibrator` — closed-loop automatic calibration with convergence detection

### Spectral Analysis

- **STFT**: Short-Time Fourier Transform with multiple window functions
- **Spectrograms**: Time-frequency magnitude/phase representations
- **MFCC**: Mel-frequency cepstral coefficients extraction
- **Power Spectrum**: Optimized FFT for power-of-2 sizes

### Wavelets

- **DWT/IDWT**: Discrete Wavelet Transform (Haar, Daubechies, Symlet, Coiflet)
- **SWT**: Stationary Wavelet Transform
- **Denoising**: Wavelet-based noise reduction

### Performance

- **Zero-copy buffers**: SharedSignalBuffer, ZeroCopyBuffer, BufferPool
- **Lock-free queues**: Thread-safe concurrent data structures
- **Ring buffers**: Real-time circular buffering with statistics
- **SIMD**: Vectorized signal operations (when enabled)
- **Async streams**: Tokio-based asynchronous stream processing

## Supported I/O

- Audio (`audio`, opt-in — links the platform audio C API): CPAL (cross-platform), JACK (Linux), ASIO (Windows)
- Network: MQTT (`mqtt`, default-on), WebSocket (`websocket`), TCP/UDP (`network`), Serial (`serial`), OSC (`osc`), ZeroMQ (`zeromq`, pure-Rust ZMTP)
- Video (`video`): FFmpeg files/network streams; camera capture via libavdevice (V4L2, DirectShow, AVFoundation) -- device *enumeration* is Linux-only in a `video`-only build. Pure-Rust alternative (`video-pure`): Y4M files and camera capture on Linux/macOS/Windows via `oximedia-capture`, no C linked; enumeration is real on every platform and preferred whenever `video-pure` is compiled in
- Files (`file`): WAV, CSV (HDF5 lives behind the `hdf5` feature — pure Rust via OxiH5)
- Optional: ROS2 bridge (requires the `ros2` feature; pure-Rust RustDDS backend, all platforms)

See [Cargo Feature Flags](#cargo-feature-flags) above for the full list.

## Testing the video backends

The `video` (FFmpeg) and `video-pure` (OxiMedia) backends are covered by a four-configuration test matrix -- both features individually, both together, and neither:

```bash
cargo nextest run -p kizzasi-io --features video
cargo nextest run -p kizzasi-io --no-default-features --features std,video-pure
cargo nextest run -p kizzasi-io --features video,video-pure
cargo nextest run -p kizzasi-io
```

`scripts/check-video-matrix.sh` (repository root) runs all four, plus `clippy -D warnings` on the
two configurations that actually compile the video code (`video,video-pure` and `video-pure`
alone) and `cargo deny check bans`, as a local stand-in for CI -- this repository's policy keeps
GitHub Actions workflows to publishing only, so this script is how the matrix gets checked
end-to-end on a development machine.

The reader-parity suite (`src/video/reader_tests.rs`) runs its assertions once per compiled-in
backend against synthetically generated Y4M fixtures -- no binary test asset needed -- plus a
scripted, deterministic `oximedia_capture::mock` backend for camera capture, so the whole suite
runs without a real video file or a real camera. Real-camera enumeration and streaming are
exercised only when this is run on a machine with a camera attached and OS permission granted;
see `TODO.md` for the state of real-device verification.

## Documentation

- [API Documentation](https://docs.rs/kizzasi-io)
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under the Apache License, Version 2.0.
