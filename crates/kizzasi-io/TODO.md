# kizzasi-io TODO

## Stream Sources

- [x] Add WebSocket stream
- [x] Implement TCP/UDP socket stream
- [x] Add serial port support (tokio-serial)
- [x] Implement file stream (WAV, CSV, HDF5)
- [x] Add ROS2 subscriber bridge (requires ROS2 installed: `ros2` feature)
- [x] Implement OSC protocol support
- [x] Add ZeroMQ stream (PULL/PUSH/PUB patterns)

## MQTT Improvements

- [x] Add TLS/SSL support
- [x] Implement QoS levels
- [x] Add retained message handling
- [x] Support wildcard topics
- [x] Implement reconnection logic
- [x] Add message batching

## Audio Improvements

- [x] Add audio output (playback)
- [x] Implement ASIO backend (Windows)
- [x] Add JACK support (Linux pro audio)
- [x] Support multi-channel audio
- [x] Implement audio file reading
- [x] Add real-time resampling (StreamingResampler, SincStreamingResampler)

## Signal Processing

- [x] Implement IIR filters (Butterworth lowpass/highpass, notch)
- [x] Add FIR filter design (sinc lowpass/highpass, moving average, differentiator)
- [x] Implement spectrogram computation (STFT with window functions)
- [x] Add MFCC extraction (Mel filterbank + DCT)
- [x] Implement wavelet transforms (DWT, IDWT, SWT, denoising with Haar, Daubechies, Symlet, Coiflet)
- [x] Add resampling (polyphase: Resampler, LinearResampler, CubicResampler)
- [x] Implement envelope detection (via Hilbert transform)

## Video Support

- [x] Add video frame input (ffmpeg) (requires FFmpeg: `video` feature)
- [x] Implement frame decimation (skip frames via config)
- [x] Add optical flow extraction (Lucas-Kanade, Dense Gradient, Block Matching)
- [x] Support camera input (v4l2/DirectShow/AVFoundation with device enumeration)
- [x] Implement frame buffering (FrameBuffer with configurable size)

## Performance

- [x] Add async stream reading (AsyncSignalStream, AsyncMemoryStream, ChannelStream)
- [x] Implement zero-copy buffers (SharedSignalBuffer, ZeroCopyBuffer, BufferPool)
- [x] Add ring buffer for real-time (RingBuffer, SignalRingBuffer with stats)
- [x] Implement lock-free queues (LockFreeQueue, UnboundedQueue, SignalQueue, LockFreeRingBuffer)
- [x] Add SIMD signal processing (rms_simd, normalize_simd, add_simd, multiply_simd with cfg(feature="simd"))
- [x] Optimize FFT for power-of-2 sizes (fft_pow2, ifft_pow2, power_spectrum_pow2, zero_pad_pow2)

## Diagnostics

- [x] Add stream health monitoring (HealthMonitor, StreamHealth, HealthStatus)
- [x] Implement latency measurement (LatencyStats, LatencyTracker with percentiles)
- [x] Add buffer underrun detection (underrun/overrun tracking, BufferLevelTracker)
- [x] Create stream statistics (AggregateHealth for multiple streams)
- [x] Add signal quality metrics (SignalQuality: SNR, clipping, crest factor)

## Testing

- [x] Add mock streams for testing (MemoryStream, AsyncMemoryStream)
- [x] Create signal generators (sine, square, sawtooth, triangle, noise, chirp, etc.)
- [x] Implement stream recording/playback (Binary, JSON, CSV formats with timestamps)
- [x] Add integration tests (25+ integration tests covering full workflows)
- [x] Add integration tests with real devices (audio backends, optical flow, camera enumeration)

## Performance & Optimization

- [x] SIMD optimizations for optical flow (auto-vectorization friendly code)
- [x] Comprehensive benchmarks suite (signal processing, optical flow, lock-free structures)
- [x] Zero-copy buffer optimizations

## Video Processing Filters

- [x] Gaussian blur for noise reduction
- [x] Box blur (simple averaging)
- [x] Sobel edge detection
- [x] Laplacian edge detection
- [x] Morphological operations (erosion, dilation)
- [x] Sharpen filter
- [x] Brightness and contrast adjustment

## Examples & Documentation

- [x] Audio processing example (filtering, FFT, spectrogram, MFCC)
- [x] Optical flow tracking example
- [x] MQTT sensor monitoring example
- [x] Real-time filtering example (low-latency processing)
- [x] Wavelet denoising example

## Advanced Features (Phase 1)

- [x] Stream processing pipeline (composable transformations)
- [x] Adaptive buffering (dynamic buffer sizing based on load)
- [x] Rate limiting and adaptive rate control
- [x] Multi-stream synchronization (time-aligned processing)
- [x] Time synchronization (NTP-like algorithm)
- [x] Phase-locked loop for stream sync
- [x] Signal transforms (scale, offset, clip, normalize, decimate, moving average, derivative)
- [x] Parallel pipeline processing with multiple combine strategies

## Advanced Features (Phase 2 - Latest)

- [x] Signal compression and decompression
  - RLE, Delta, Delta+RLE, Quantization, DPCM
  - Adaptive compression (automatic method selection)
  - Lossless and lossy compression support
- [x] Sensor calibration utilities
  - Multi-point calibration with least squares
  - Non-linear calibration curves
  - Temperature compensation
  - Auto-calibration with reference signals
  - Calibration management for multiple sensors
  - JSON serialization/deserialization
- [x] Batch processing utilities
  - Configurable batch accumulation
  - Time-based and size-based flushing
  - Windowed batch processing with overlap
  - Parallel batch processing (multi-threaded)
  - Batch statistics tracking
- [x] Stream multiplexing and demultiplexing
  - Multiple merge strategies (round-robin, time-ordered, weighted)
  - Stream splitting with custom routing
  - Async channel-based multiplexing
  - Broadcast to multiple outputs

## Advanced Features (Phase 3 - State-of-the-Art Signal Processing)

- [x] Advanced adaptive filtering
  - Kalman Filter for optimal linear state estimation
  - Extended Kalman Filter support (matrix operations)
  - Particle Filter for non-Gaussian/nonlinear systems
  - LMS (Least Mean Squares) adaptive filter
  - NLMS (Normalized LMS) for faster convergence
  - RLS (Recursive Least Squares) for rapid adaptation
  - Comprehensive state estimation and tracking

- [x] Cepstral analysis and speech processing
  - Real cepstrum computation for pitch detection
  - Complex cepstrum for homomorphic deconvolution
  - Liftering (cepstral windowing) operations
  - Formant tracking and speech analysis
  - Pitch detection using quefrency domain
  - Cepstral distance metrics for quality assessment
  - Quefrency domain filtering
  - Minimum-phase signal reconstruction
  - Phase unwrapping algorithms

- [x] Advanced time-frequency analysis
  - Gabor transform with Gaussian windows
  - Gabor synthesis (inverse transform)
  - S-transform (Stockwell) for frequency-dependent resolution
  - Wigner-Ville distribution for high-resolution TF analysis
  - Choi-Williams distribution with cross-term suppression
  - Reassigned spectrogram for improved resolution
  - Multiple window function support

## Advanced Features (Phase 4 - Multi-channel & Source Separation)

- [x] Blind source separation
  - FastICA (Fast Independent Component Analysis) with multiple nonlinearities
  - Non-negative Matrix Factorization (NMF) with multiplicative updates
  - Principal Component Analysis (PCA) for dimensionality reduction
  - Temporal decorrelation for time-domain separation
  - Full eigenvalue decomposition implementation
  - Gram-Schmidt orthogonalization

- [x] Multi-channel audio processing
  - Delay-and-Sum beamforming for spatial filtering
  - MVDR (Minimum Variance Distortionless Response) beamforming
  - Direction of Arrival (DOA) estimation using SRP
  - Adaptive beamforming with LMS algorithm
  - Microphone array configurations (linear, circular)
  - Steering vector computation
  - Spatial covariance estimation

## Completed Enhancements (Phase 5)

- [x] Hilbert-Huang Transform (HHT)
  - Empirical Mode Decomposition (EMD) with sifting process
  - Ensemble EMD (EEMD) for noise-assisted decomposition
  - Intrinsic Mode Functions (IMFs) extraction
  - Hilbert spectral analysis with instantaneous amplitude/frequency/phase
  - Instantaneous frequency tracking via Hilbert transform
  - Cubic spline interpolation for envelope construction

- [x] Signal quality metrics
  - PESQ-inspired perceptual quality metric (1.0-4.5 scale)
  - POLQA-inspired quality assessment (1.0-5.0 scale)
  - STOI (Short-Time Objective Intelligibility) (0.0-1.0 scale)
  - Mean Opinion Score (MOS) prediction (1.0-5.0 scale)
  - SNR and segmental SNR calculation
  - Frequency-weighted SNR with A-weighting
  - Comprehensive quality metrics with ratings

- [x] Advanced resampling techniques
  - Rational resampling (L/M) with polyphase filterbank (already implemented as `Resampler`)
  - Arbitrary sample rate conversion (`ArbitrarySrcResampler`)
  - Farrow structure for fractional delay (`FarrowResampler`)
  - Polynomial interpolation (linear, cubic Lagrange)
  - Time-varying resampling (`TimeVaryingResampler`)
  - Multiple modulation types (sinusoidal, linear/exponential chirp)
  - Dynamic ratio adjustment

- [ ] Machine learning integration
  - Neural network-based denoising
  - Feature extraction for ML pipelines
  - Autoencoder-based compression
  - Anomaly detection in signals
  - Transfer learning support

## Testing and Documentation

- [x] Unit tests (128+ tests across all modules)
- [x] Integration tests (24 comprehensive workflow tests)
- [x] Doc tests (13 documentation examples)
- [x] Source separation tests (FastICA, NMF, PCA)
- [x] Beamforming tests (Delay-and-Sum, MVDR, DOA)
- [x] HHT tests (EMD, EEMD, extrema detection, Hilbert transform)
- [x] Quality metrics tests (SNR, STOI, PESQ, POLQA, MOS)
- [x] Advanced resampling tests (Farrow, time-varying, arbitrary SRC)
- [x] Usage examples
  - Audio processing example
  - Optical flow tracking example
  - MQTT sensor monitoring example
  - Real-time filtering example
  - Wavelet denoising example
  - Hilbert-Huang Transform example
  - Source separation example (created)
  - Beamforming example (created)
- [x] Performance benchmarks for Phase 3-5 features
  - HHT (EMD, EEMD)
  - Quality metrics (SNR, PESQ, STOI, POLQA, MOS)
  - Advanced resampling (Farrow, time-varying, arbitrary SRC)
  - Adaptive filters (Kalman, LMS, NLMS, RLS, Particle)
  - Cepstral analysis (Real/Complex cepstrum, formant tracking)
  - Time-frequency analysis (Gabor, S-transform, Wigner-Ville, Choi-Williams, Reassigned)
  - Source separation (FastICA, NMF, PCA, Temporal decorrelation)
  - Beamforming (Delay-and-Sum, MVDR, Adaptive, DOA)
  - Note: Benchmark framework in place; some benchmarks may need API refinement
- [ ] Tutorial documentation for advanced features
- [ ] API reference updates
