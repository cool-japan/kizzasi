//! # kizzasi-io
//!
//! Physical world connectors for Kizzasi - MQTT, Audio, Video, and comprehensive sensor streams
//! with state-of-the-art signal processing capabilities.
//!
//! ## Core Features
//!
//! ### Stream Sources
//! - **MQTT**: IoT/industrial sensor integration with TLS, QoS, and reconnection
//! - **Audio**: Multi-channel audio I/O via cpal (ASIO, JACK support)
//! - **Video**: Frame capture with FFmpeg and camera input (v4l2/DirectShow/AVFoundation)
//! - **Network**: WebSocket, TCP/UDP sockets, serial ports
//! - **Protocols**: OSC, ZeroMQ, ROS2 (when available)
//! - **Files**: WAV, CSV, HDF5 reading/writing
//!
//! ### Signal Processing
//!
//! #### Basic Filters
//! - **FIR**: Sinc lowpass/highpass, moving average, differentiator
//! - **IIR**: Butterworth lowpass/highpass, notch filters
//!
//! #### Advanced Adaptive Filtering
//! - **Kalman Filter**: Optimal linear state estimation with prediction/update cycles
//! - **Particle Filter**: Non-Gaussian/nonlinear Bayesian estimation
//! - **LMS/NLMS**: Least Mean Squares adaptive filtering
//! - **RLS**: Recursive Least Squares for rapid adaptation
//!
//! #### Cepstral Analysis
//! - **Real Cepstrum**: Pitch detection and formant analysis
//! - **Complex Cepstrum**: Homomorphic deconvolution
//! - **Formant Tracking**: Speech resonance detection
//! - **Liftering**: Quefrency domain filtering
//! - **Cepstral Distance**: Speech quality assessment
//!
//! #### Time-Frequency Analysis
//! - **Gabor Transform**: Optimal time-frequency resolution with Gaussian windows
//! - **S-Transform**: Frequency-dependent resolution (Stockwell transform)
//! - **Wigner-Ville**: High-resolution quadratic distribution
//! - **Choi-Williams**: Cross-term suppression for multi-component signals
//! - **Reassigned Spectrogram**: Enhanced resolution through energy reassignment
//!
//! #### Spectral Analysis
//! - **STFT**: Short-Time Fourier Transform with multiple window functions
//! - **Spectrograms**: Time-frequency magnitude/phase representations
//! - **MFCC**: Mel-frequency cepstral coefficients extraction
//! - **Power Spectrum**: Optimized FFT for power-of-2 sizes
//!
//! #### Wavelets
//! - **DWT/IDWT**: Discrete Wavelet Transform (Haar, Daubechies, Symlet, Coiflet)
//! - **SWT**: Stationary Wavelet Transform
//! - **Denoising**: Wavelet-based noise reduction
//!
//! #### Resampling
//! - **Polyphase**: Efficient integer-ratio resampling
//! - **Linear/Cubic**: Interpolation-based resampling
//! - **Streaming**: Real-time sinc-based resampling
//!
//! ### Performance Optimization
//! - **Zero-copy buffers**: SharedSignalBuffer, ZeroCopyBuffer, BufferPool
//! - **Lock-free queues**: Thread-safe concurrent data structures
//! - **Ring buffers**: Real-time circular buffering with statistics
//! - **SIMD**: Vectorized signal operations (when enabled)
//! - **Async streams**: Tokio-based asynchronous stream processing
//!
//! ### Stream Processing
//! - **Pipelines**: Composable signal transformations
//! - **Synchronization**: Multi-stream time alignment with PLL
//! - **Multiplexing**: Round-robin, time-ordered, weighted merging
//! - **Adaptive buffering**: Dynamic buffer sizing
//! - **Rate control**: Adaptive rate limiting
//!
//! ### Diagnostics & Monitoring
//! - **Health monitoring**: Stream status, latency, throughput
//! - **Signal quality**: SNR, clipping detection, crest factor
//! - **Latency tracking**: Percentile-based latency statistics
//! - **Buffer tracking**: Underrun/overrun detection
//!
//! ### Data Processing
//! - **Compression**: RLE, Delta, DPCM, Quantization with adaptive selection
//! - **Calibration**: Multi-point, non-linear, temperature compensation
//! - **Batch processing**: Windowed, parallel batch operations
//! - **Recording/Playback**: Binary, JSON, CSV stream capture
//!
//! ### Video Processing
//! - **Optical Flow**: Lucas-Kanade, Dense Gradient, Block Matching
//! - **Filters**: Gaussian/Box blur, Sobel/Laplacian edges, morphology
//! - **Frame operations**: Format conversion, normalization, buffering
//!
//! ## Examples
//!
//! ### Basic Signal Generation and Filtering
//! ```rust,no_run
//! use kizzasi_io::{SineGenerator, SignalGenerator, FirFilter};
//! use scirs2_core::ndarray::Array1;
//!
//! // Generate a sine wave
//! let mut generator = SineGenerator::new(440.0, 1.0, 44100.0);
//! let signal = generator.generate(1024);
//!
//! // Apply lowpass filter
//! let mut filter = FirFilter::sinc_lowpass(0.25, 31).unwrap();
//! let filtered = filter.process(&signal);
//! ```
//!
//! ### Adaptive Filtering with Kalman Filter
//! ```rust,no_run
//! use kizzasi_io::KalmanFilter;
//! use scirs2_core::ndarray::{arr1, Array2};
//!
//! // 1D position tracking
//! let initial_state = arr1(&[0.0]);
//! let initial_cov = Array2::eye(1);
//! let transition = Array2::eye(1);
//! let observation = Array2::eye(1);
//! let process_noise = Array2::eye(1) * 0.01;
//! let measurement_noise = Array2::eye(1) * 0.1;
//!
//! let mut kf = KalmanFilter::new(
//!     initial_state, initial_cov, transition,
//!     observation, process_noise, measurement_noise
//! ).unwrap();
//!
//! // Update with measurement
//! kf.predict();
//! kf.update(&arr1(&[1.5])).unwrap();
//! println!("State estimate: {:?}", kf.state());
//! ```
//!
//! ### Time-Frequency Analysis
//! ```rust,no_run
//! use kizzasi_io::{GaborTransform, SineGenerator, SignalGenerator};
//!
//! let mut gen = SineGenerator::new(440.0, 1.0, 16000.0);
//! let signal = gen.generate(1024);
//!
//! let mut gabor = GaborTransform::new(16000.0);
//! let result = gabor.compute(&signal, 256, 128, 32.0).unwrap();
//!
//! println!("Time-frequency representation: {} × {} bins",
//!          result.num_frames, result.num_bins);
//! ```
//!
//! ## COOLJAPAN Ecosystem
//!
//! This crate uses `scirs2-core` for all array operations following KIZZASI_POLICY.md.

mod adaptive;
mod batch;
mod calibration;
mod compression;
mod error;
mod generator;
mod health;
mod lockfree;
mod multiplex;
mod pipeline;
mod recorder;
mod signal;
mod stream;
mod sync;
mod zerocopy;

#[cfg(feature = "mqtt")]
mod mqtt;

#[cfg(feature = "audio")]
mod audio;

#[cfg(feature = "websocket")]
mod websocket;

#[cfg(feature = "serial")]
mod serial;

#[cfg(feature = "file")]
mod file;

#[cfg(feature = "network")]
mod socket;

#[cfg(feature = "osc")]
mod osc;

#[cfg(feature = "zeromq")]
mod zeromq;

#[cfg(feature = "ros2")]
mod ros2;

#[cfg(any(feature = "video", feature = "video-pure"))]
mod video;

pub use adaptive::{
    AdaptiveBuffer, AdaptiveBufferStats, AdaptiveConfig, AdaptiveRateController, AdaptiveStrategy,
    RateLimiter,
};
pub use batch::{
    BatchAccumulator, BatchConfig, BatchProcessor, BatchStats, ParallelBatchProcessor,
    WindowedBatchProcessor,
};
pub use calibration::{
    AutoCalibrator, CalibrationCurve, CalibrationManager, CalibrationParams, MultiPointCalibrator,
};
pub use compression::{
    AdaptiveCompressor, CompressedSignal, CompressionMetadata, CompressionMethod, SignalCompressor,
};
pub use error::{IoError, IoResult};
pub use generator::{
    ChirpGenerator, ImpulseGenerator, MultiToneGenerator, PinkNoiseGenerator, SawtoothGenerator,
    SignalGenerator, SineGenerator, SquareGenerator, StepGenerator, TriangleGenerator,
    WhiteNoiseGenerator,
};
pub use health::{
    AggregateHealth, BufferLevelTracker, HealthMonitor, HealthStatus, LatencyStats, LatencyTracker,
    SignalQuality, StreamHealth,
};
pub use lockfree::{LockFreeQueue, LockFreeRingBuffer, SignalQueue, UnboundedQueue};
pub use multiplex::{
    AsyncMultiplexer, ChannelSplitter, MultiplexConfig, MultiplexStrategy, StreamDemultiplexer,
    StreamMultiplexer,
};
pub use pipeline::{
    ClipTransform, CombineStrategy, DecimateTransform, DerivativeTransform, MovingAverageTransform,
    NormalizeTransform, OffsetTransform, ParallelPipeline, Pipeline, ScaleTransform,
    StreamTransform,
};
pub use recorder::{RecordedFrame, RecorderConfig, RecorderFormat, StreamPlayer, StreamRecorder};
pub use signal::{
    AdaptiveBeamformer, ArbitrarySrcResampler, CepstralDistance, ChoiWilliams, ComplexCepstrum,
    CubicResampler, DOAEstimator, DelayAndSum, DwtMultiLevel, DwtResult, EmdConfig, EmdResult,
    EmpiricalModeDecomposition, EnsembleEmd, FarrowResampler, FastICA, Filter, FirFilter,
    FormantTracker, GaborResult, GaborTransform, IirFilter, IntrinsicModeFunction, KalmanFilter,
    LinearResampler, LmsFilter, MicrophoneArray, MosPredictor, NlmsFilter, Nonlinearity,
    ParticleFilter, PesqCalculator, PolqaCalculator, QualityMetrics, QuefrencyFilter,
    RatioModulation, RealCepstrum, ReassignedResult, ReassignedSpectrogram, Resampler, RlsFilter,
    STransform, STransformResult, SignalProcessor, SincStreamingResampler, SnrCalculator,
    Spectrogram, StoiCalculator, StreamingResampler, TemporalDecorrelation, TimeVaryingResampler,
    WaveletAnalyzer, WaveletType, WignerVille, WignerVilleResult, WindowType, MVDR, NMF, PCA,
};
pub use stream::{
    AsyncMemoryStream, AsyncSignalStream, ChannelStream, MemoryStream, RingBuffer, RingBufferIter,
    SignalRingBuffer, SignalStream, StreamConfig,
};
pub use sync::{
    InterpolationMethod, PhaseLockLoop, StreamSynchronizer, SyncConfig, TimeSynchronizer,
    Timestamp, TimestampedSample,
};
pub use zerocopy::{BufferPool, SharedSignalBuffer, ZeroCopyBuffer, ZeroCopyBufferMut};

// Re-export scirs2-core types
pub use scirs2_core::ndarray::Array1;

#[cfg(feature = "mqtt")]
pub use mqtt::{MqttClient, MqttConfig, MqttMessage, MqttStream, QosLevel, TlsConfig};

#[cfg(feature = "audio")]
pub use audio::{AudioBackend, AudioConfig, AudioInput, AudioOutput};

#[cfg(feature = "websocket")]
pub use websocket::{WebSocketConfig, WebSocketStream};

#[cfg(feature = "serial")]
pub use serial::{list_ports, DataBits, FlowControl, Parity, SerialConfig, SerialStream, StopBits};

#[cfg(feature = "file")]
pub use file::{CsvReader, CsvWriter, SampleFormat, WavReader, WavSpec, WavWriter};

// HDF5 is pure Rust via OxiH5 (no libhdf5 install needed); it stays a
// separate opt-in feature rather than part of `file` only because HDF5
// support is optional (see Cargo.toml).
#[cfg(feature = "hdf5")]
pub use file::{Hdf5Reader, Hdf5Writer};

#[cfg(feature = "network")]
pub use socket::{SocketConfig, TcpClientStream, TcpServerStream, UdpSocketStream};

#[cfg(feature = "osc")]
pub use osc::{OscArg, OscMessage, OscReceiver, OscSender, OscServer};

#[cfg(feature = "zeromq")]
pub use zeromq::{ZmqConfig, ZmqMessage, ZmqPattern, ZmqStream};

#[cfg(feature = "ros2")]
pub use ros2::{
    imu_to_samples, msg as ros2_msg, QosProfile, Ros2Config, Ros2MessageType, Ros2Stream,
};

#[cfg(any(feature = "video", feature = "video-pure"))]
pub use video::{
    CameraDevice, FrameBuffer, OpticalFlow, OpticalFlowEstimator, OpticalFlowMethod, PixelFormat,
    VideoBackend, VideoConfig, VideoFilter, VideoFrame, VideoMetadata, VideoProcessor, VideoReader,
    VideoSource,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_processor() {
        let processor = SignalProcessor::new(1024);
        assert_eq!(processor.buffer_size(), 1024);
    }
}
