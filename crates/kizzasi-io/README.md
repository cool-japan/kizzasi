# kizzasi-io

Physical world connectors for Kizzasi - MQTT, Audio, and sensor streams.

## Overview

Comprehensive I/O toolkit for real-time signal acquisition and processing. Connects Kizzasi to sensors, audio devices, network protocols, and file formats.

## Features

- **MQTT**: TLS, QoS, wildcard topics, reconnection logic
- **Audio**: CPAL, JACK, ASIO backends with multi-channel support
- **Video**: FFmpeg integration, optical flow, camera input
- **Signal Processing**: FFT, filters (IIR/FIR), wavelets, MFCC extraction
- **Network**: WebSocket, TCP/UDP, serial, OSC, ZeroMQ
- **File I/O**: WAV, CSV, HDF5 support
- **Advanced DSP**: Hilbert-Huang Transform, beamforming, source separation
- **Performance**: Lock-free queues, zero-copy buffers, SIMD operations

## Quick Start

```rust
use kizzasi_io::{AudioInput, StreamConfig, SignalProcessor};

// Audio input at 16kHz
let config = StreamConfig::new(16000, 1)?; // sample_rate, channels
let mut audio = AudioInput::new(config)?;

// Read audio samples
let samples = audio.read(1024)?;

// Apply filtering
let mut processor = SignalProcessor::new(16000);
let filtered = processor.butterworth_lowpass(&samples, 1000.0)?;

// MQTT streaming
use kizzasi_io::MqttClient;
let mut client = MqttClient::new("mqtt://broker.local", "sensor")?;
client.publish("readings/temperature", &data).await?;
```

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
- `DOAEstimator` — Direction-of-Arrival estimation (MUSIC, GCC-PHAT)

### Speech Quality Metrics

- `PesqCalculator` — PESQ (ITU-T P.862) perceptual evaluation of speech quality
- `StoiCalculator` — STOI (Short-Time Objective Intelligibility) scoring
- `PolqaCalculator` — POLQA (ITU-T P.863) wideband quality measurement
- `MosPredictor` — Mean Opinion Score prediction from signal features

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

- Audio: CPAL (cross-platform), JACK (Linux), ASIO (Windows)
- Network: MQTT, WebSocket, TCP/UDP, Serial, OSC, ZeroMQ
- Video: FFmpeg, V4L2, DirectShow, AVFoundation
- Files: WAV, CSV, HDF5
- Optional: ROS2 bridge (requires `ros2` feature)
- 198 comprehensive tests, all passing

## Documentation

- [API Documentation](https://docs.rs/kizzasi-io)
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under the Apache License, Version 2.0.
