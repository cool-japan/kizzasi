//! Integration tests for kizzasi-io
//!
//! Tests full workflows with mock devices and real-world scenarios

use kizzasi_io::*;
use scirs2_core::ndarray::Array1;
use std::time::Duration;

#[test]
fn test_signal_chain_processing() {
    // Create a signal generator
    let mut generator = SineGenerator::new(440.0, 44100.0, 0.5);

    // Generate a signal
    let signal = generator.generate(1024);

    // Process through signal processor
    let mut processor = SignalProcessor::new(1024);
    let fft_result = processor.fft(&signal).unwrap();

    // Should have correct length
    assert_eq!(fft_result.len(), 1024);

    // Check that FFT has significant magnitude at 440 Hz
    // 440 Hz / (44100 / 1024) ≈ 10.2 bins
    let bin = (440.0 * 1024.0 / 44100.0) as usize;
    assert!(fft_result[bin].norm() > 100.0);
}

#[test]
fn test_memory_stream_to_ring_buffer() {
    // Create test data
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let config = StreamConfig {
        sample_rate: 44100.0,
        channels: 1,
        buffer_size: 4,
        timeout: Some(Duration::from_secs(1)),
    };

    // Create memory stream
    let mut stream = MemoryStream::new(data.clone(), config.clone());

    // Create ring buffer
    let mut ring_buffer = SignalRingBuffer::new(10);

    // Read from stream and push to ring buffer
    while stream.is_active() {
        let buf = stream.read().expect("MemoryStream::read should succeed");
        ring_buffer.push_slice(buf.as_slice().unwrap());
    }

    // Verify data was transferred correctly
    assert_eq!(ring_buffer.len(), 8);
    assert!((ring_buffer.mean() - 4.5).abs() < 0.1);
}

#[test]
fn test_signal_generator_pipeline() {
    // Generate multiple signal types
    let sample_rate = 44100.0;
    let samples = 1024;

    let mut sine_gen = SineGenerator::new(1000.0, sample_rate, 0.5);
    let sine_signal = sine_gen.generate(samples);

    let mut square_gen = SquareGenerator::new(500.0, sample_rate, 0.3);
    let square_signal = square_gen.generate(samples);

    // Mix signals (simple addition)
    let mut mixed = Vec::with_capacity(samples);
    for i in 0..samples {
        mixed.push(sine_signal[i] + square_signal[i]);
    }
    let mixed_array = Array1::from_vec(mixed);

    // Apply filtering
    let mut processor = SignalProcessor::new(samples);
    let filtered = processor
        .apply_filter(
            &mixed_array,
            Filter::LowPass {
                cutoff: 0.1,
                order: 4,
            },
        )
        .unwrap();

    // Filtered signal should be different from original
    assert_ne!(filtered[0], mixed_array[0]);
    assert_eq!(filtered.len(), samples);
}

#[test]
fn test_health_monitoring_workflow() {
    // Create health monitor
    let mut monitor = HealthMonitor::new();

    // Simulate processing some samples
    let dummy_samples = vec![0.0f32; 1024];
    for i in 0..100 {
        monitor.record_samples(&dummy_samples);
        monitor.record_latency(Duration::from_micros(500 + i * 10));

        // Simulate occasional underrun
        if i % 20 == 0 {
            monitor.record_underrun();
        }
    }

    // Get health status
    let health = monitor.health();

    // Should have processed samples
    assert_eq!(health.samples_processed, 1024 * 100);

    // Should have detected underruns
    assert_eq!(health.underruns, 5);
}

#[test]
fn test_lock_free_queue_concurrent_access() {
    // Create a lock-free queue
    let queue = LockFreeQueue::new(128);

    // Push items
    for i in 0..100 {
        assert!(queue.try_push(i as f32));
    }

    // Pop items
    let mut sum = 0.0;
    let mut count = 0;
    while let Some(val) = queue.pop() {
        sum += val;
        count += 1;
    }

    assert_eq!(count, 100);
    assert_eq!(sum, (0..100).sum::<i32>() as f32);
}

#[test]
fn test_spectrogram_analysis() {
    // Generate a chirp signal
    let mut generator = ChirpGenerator::new(100.0, 2000.0, 44100.0, 2.0, 0.7);
    let signal = generator.generate(88200); // 2 seconds at 44.1 kHz

    // Compute spectrogram
    let mut processor = SignalProcessor::new(88200);
    let spec = processor
        .spectrogram(&signal, 2048, 512, WindowType::Hann)
        .unwrap();

    // Should have multiple time frames
    assert!(spec.num_frames > 100);

    // Should have frequency bins
    assert_eq!(spec.num_bins, 1025); // 2048/2 + 1

    // Convert to dB
    let db_spec = spec.to_db(1.0, -80.0);
    assert!(!db_spec.is_empty());
}

#[test]
fn test_signal_resampling_chain() {
    // Generate signal at 44.1 kHz
    let mut generator = SineGenerator::new(1000.0, 44100.0, 0.5);
    let signal = generator.generate(4410); // 100ms

    // Resample to 48 kHz
    let mut resampler = StreamingResampler::new(44100.0, 48000.0);
    let resampled = resampler.process(&signal.to_vec());

    // Should have approximately the right length
    // 4410 samples at 44.1kHz -> ~4800 samples at 48kHz
    assert!(resampled.len() > 4700 && resampled.len() < 4900);

    // Signal should not be all zeros
    let sum: f32 = resampled.iter().map(|x| x.abs()).sum();
    assert!(sum > 100.0);
}

#[test]
fn test_iir_filter_stability() {
    // Create a Butterworth lowpass filter
    let mut filter = IirFilter::butterworth_lowpass(0.1).unwrap();

    // Process a step function
    let step = Array1::from_vec(
        vec![0.0; 100]
            .into_iter()
            .chain(vec![1.0; 100])
            .collect::<Vec<_>>(),
    );

    let output = filter.process(&step);

    // Output should be stable (not diverge)
    assert!(output.iter().all(|&x| x.abs() < 10.0));

    // Output should respond to the step
    assert!(output[150] > 0.1);
}

#[test]
fn test_wavelet_analysis() {
    // Generate test signal
    let mut sine_gen = SineGenerator::new(100.0, 8000.0, 0.8);
    let signal = sine_gen.generate(1024);

    // Perform wavelet decomposition
    let analyzer = WaveletAnalyzer::new(WaveletType::Daubechies4);
    let result = analyzer.dwt(signal.as_slice().unwrap());

    // Should have approximation and detail coefficients
    assert!(!result.approximation.is_empty());
    assert!(!result.detail.is_empty());
    assert_eq!(result.level, 1); // dwt performs single level decomposition

    // The transform must actually be invertible. This assertion is what was
    // missing while `idwt` silently returned samples unrelated to the input
    // for every wavelet except Haar.
    let samples = signal.as_slice().unwrap();
    let reconstructed = analyzer.idwt(&result.approximation, &result.detail, samples.len());
    assert_eq!(reconstructed.len(), samples.len());
    let max_error = samples
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error < 1e-4, "DWT round trip error {max_error}");

    // ... and so must the multi-level transform used by `denoise`.
    let multi = analyzer.dwt_multilevel(samples, 3);
    let rebuilt = analyzer.idwt_multilevel(&multi);
    let multi_error = samples
        .iter()
        .zip(rebuilt.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        multi_error < 1e-3,
        "multi-level round trip error {multi_error}"
    );
}

#[test]
fn test_zero_copy_buffer_chain() {
    // Create zero-copy buffer
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let buffer = SharedSignalBuffer::new(data.clone());

    // Create multiple slices (zero-copy)
    let slice1 = buffer.slice(0, 3);
    let slice2 = buffer.slice(2, 5);

    // Should have correct data
    assert_eq!(slice1.as_slice(), &[1.0, 2.0, 3.0]);
    assert_eq!(slice2.as_slice(), &[3.0, 4.0, 5.0]);

    // Should share the same underlying data
    assert_eq!(buffer.ref_count(), 3); // original + 2 slices
}

#[tokio::test]
async fn test_async_stream_processing() {
    // Create async memory stream
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let config = StreamConfig::default();
    let mut stream = AsyncMemoryStream::new(data.clone(), config);

    // Read asynchronously. Per the stream read contract the buffer holds
    // ONLY real samples: an 8-sample source yields an 8-sample block, not a
    // 1024-sample block zero-padded with fabricated silence (which is what
    // this test used to assert).
    let chunk = stream.read().await.unwrap();

    assert_eq!(chunk.len(), data.len());
    assert_eq!(chunk[0], 1.0);
    assert_eq!(chunk[7], 8.0);

    // Once exhausted the stream reports EndOfStream rather than zeros.
    assert!(matches!(
        stream.read().await,
        Err(kizzasi_io::IoError::EndOfStream)
    ));
}

#[tokio::test]
async fn test_recorder_config_creation() {
    // Test recorder configuration creation
    let temp_dir = std::env::temp_dir();
    let rec_path = temp_dir.join("test_recording.bin");

    let config = RecorderConfig {
        path: rec_path.to_str().unwrap().to_string(),
        format: RecorderFormat::Binary,
        sample_rate: 44100.0,
        channels: 1,
        buffer_size: 1024,
        metadata: std::collections::HashMap::new(),
        record_timestamps: false,
    };

    // Verify config
    assert_eq!(config.sample_rate, 44100.0);
    assert_eq!(config.channels, 1);
    assert_eq!(config.buffer_size, 1024);
    assert_eq!(config.format, RecorderFormat::Binary);
}

#[cfg(feature = "zeromq")]
#[test]
fn test_zmq_message_construction() {
    use kizzasi_io::{ZmqConfig, ZmqMessage, ZmqPattern};

    // Create a message
    let msg = ZmqMessage::with_topic("sensor/temperature", &b"25.5"[..]);

    assert_eq!(msg.topic, Some("sensor/temperature".to_string()));
    assert_eq!(msg.payload.len(), 4);

    // Create config
    let config = ZmqConfig {
        endpoint: "tcp://localhost:5555".to_string(),
        pattern: ZmqPattern::Pub,
        ..Default::default()
    };

    assert_eq!(config.pattern, ZmqPattern::Pub);
}

#[cfg(feature = "ros2")]
#[test]
fn test_ros2_config_creation() {
    use kizzasi_io::{QosProfile, Ros2Config, Ros2MessageType};

    // Create a basic ROS2 config
    let config = Ros2Config::new("/imu/data", Ros2MessageType::Imu);

    assert_eq!(config.topic, "/imu/data");
    assert_eq!(config.message_type, Ros2MessageType::Imu);
    assert_eq!(config.qos, QosProfile::SensorData);
    assert!(config.node_name.contains("imu_data"));

    // Test builder pattern
    let config = Ros2Config::new("/laser_scan", Ros2MessageType::LaserScan)
        .with_qos(QosProfile::SystemDefault)
        .with_buffer_size(2048)
        .with_sample_rate(50.0)
        .with_channels(3);

    assert_eq!(config.buffer_size, 2048);
    assert_eq!(config.sample_rate, 50.0);
    assert_eq!(config.channels, 3);
    assert_eq!(config.qos, QosProfile::SystemDefault);
}

#[cfg(feature = "ros2")]
#[test]
fn test_ros2_message_types() {
    use kizzasi_io::Ros2MessageType;

    let msg_types = vec![
        Ros2MessageType::Float32,
        Ros2MessageType::Float64,
        Ros2MessageType::Float32Array,
        Ros2MessageType::Float64Array,
        Ros2MessageType::Imu,
        Ros2MessageType::LaserScan,
    ];

    // Verify message types can be created and compared
    for msg_type in msg_types {
        assert_eq!(msg_type.clone(), msg_type);
    }
}

#[cfg(feature = "audio")]
#[test]
fn test_audio_backend_configuration() {
    use kizzasi_io::{AudioBackend, AudioConfig};

    // Test default backend
    let config = AudioConfig::new().sample_rate(48000).channels(2);
    assert_eq!(config.backend, AudioBackend::Default);
    assert_eq!(config.sample_rate, 48000);
    assert_eq!(config.channels, 2);

    // Test backend selection
    #[cfg(target_os = "windows")]
    {
        let asio_config = AudioConfig::new().backend(AudioBackend::Asio);
        assert_eq!(asio_config.backend, AudioBackend::Asio);
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let jack_config = AudioConfig::new().backend(AudioBackend::Jack);
        assert_eq!(jack_config.backend, AudioBackend::Jack);
    }
}

#[cfg(feature = "audio")]
#[test]
fn test_audio_device_listing() {
    use kizzasi_io::{AudioInput, AudioOutput};

    // List input devices
    let input_devices = AudioInput::list_devices();
    assert!(input_devices.is_ok());

    // List output devices
    let output_devices = AudioOutput::list_devices();
    assert!(output_devices.is_ok());

    // Should have at least one device on most systems
    // (This might fail on headless CI systems, so we just check the call succeeds)
}

#[cfg(any(feature = "video", feature = "video-pure"))]
#[test]
fn test_optical_flow_computation() {
    use kizzasi_io::{OpticalFlowEstimator, OpticalFlowMethod, VideoFrame};

    // Create two test frames with slight motion
    let width = 64;
    let height = 64;
    let mut frame1_data = vec![0u8; width * height];
    let mut frame2_data = vec![0u8; width * height];

    // Create a simple pattern in frame1
    for y in 20..40 {
        for x in 20..40 {
            frame1_data[y * width + x] = 255;
        }
    }

    // Shift the pattern slightly in frame2 (simulate motion)
    for y in 20..40 {
        for x in 22..42 {
            // Shifted 2 pixels to the right
            frame2_data[y * width + x] = 255;
        }
    }

    let frame1 = VideoFrame {
        index: 0,
        timestamp: 0.0,
        width,
        height,
        channels: 1,
        data: frame1_data,
    };

    let frame2 = VideoFrame {
        index: 1,
        timestamp: 0.033,
        width,
        height,
        channels: 1,
        data: frame2_data,
    };

    // Test different optical flow methods
    let methods = vec![
        OpticalFlowMethod::DenseGradient,
        OpticalFlowMethod::BlockMatching,
        OpticalFlowMethod::LucasKanade,
    ];

    for method in methods {
        let estimator = OpticalFlowEstimator::new(method).with_window_size(9);
        let flow = estimator.compute(&frame1, &frame2).unwrap();

        // Check flow dimensions
        assert_eq!(flow.dimensions(), (height, width));

        // Flow should detect some motion
        let max_magnitude = flow.max_magnitude();
        assert!(max_magnitude > 0.0);
    }
}

#[cfg(any(feature = "video", feature = "video-pure"))]
#[test]
fn test_optical_flow_properties() {
    use kizzasi_io::{OpticalFlowEstimator, OpticalFlowMethod, VideoFrame};

    // Create identical frames (no motion)
    let width = 32;
    let height = 32;
    let data = vec![128u8; width * height];

    let frame1 = VideoFrame {
        index: 0,
        timestamp: 0.0,
        width,
        height,
        channels: 1,
        data: data.clone(),
    };

    let frame2 = VideoFrame {
        index: 1,
        timestamp: 0.033,
        width,
        height,
        channels: 1,
        data,
    };

    // Compute flow
    let estimator = OpticalFlowEstimator::new(OpticalFlowMethod::DenseGradient);
    let flow = estimator.compute(&frame1, &frame2).unwrap();

    // With identical frames, flow should be near zero
    let avg_magnitude = flow.avg_magnitude();
    assert!(avg_magnitude < 1.0);

    // Test flow vector access
    let flow_vec = flow.get_flow(width / 2, height / 2);
    assert!(flow_vec.is_some());

    // Test out-of-bounds access
    let out_of_bounds = flow.get_flow(width + 10, height + 10);
    assert!(out_of_bounds.is_none());
}

#[cfg(any(feature = "video", feature = "video-pure"))]
#[test]
fn test_camera_device_enumeration() {
    use kizzasi_io::CameraDevice;

    // Get default camera format for platform
    let default_format = CameraDevice::default_format();
    assert!(!default_format.is_empty());

    #[cfg(target_os = "linux")]
    assert_eq!(default_format, "video4linux2");

    #[cfg(target_os = "windows")]
    assert_eq!(default_format, "dshow");

    #[cfg(target_os = "macos")]
    assert_eq!(default_format, "avfoundation");

    // List available devices (may fail if no cameras present)
    let devices_result = CameraDevice::list_devices();
    // We don't assert this succeeds, as CI environments may not have cameras
    // Just verify the call doesn't panic
    match devices_result {
        Ok(devices) => {
            for device in devices {
                assert!(!device.path.is_empty());
                assert!(!device.formats.is_empty());
            }
        }
        Err(_) => {
            // No cameras available, which is fine for testing
        }
    }
}

#[cfg(any(feature = "video", feature = "video-pure"))]
#[test]
fn test_video_config_camera() {
    use kizzasi_io::{VideoConfig, VideoSource};

    // Test camera configuration
    let config = VideoConfig::from_camera("/dev/video0")
        .with_camera_fps(60)
        .with_camera_format("video4linux2")
        .with_resize(1280, 720);

    assert_eq!(config.camera_fps, Some(60));
    assert_eq!(config.camera_format, Some("video4linux2".to_string()));
    assert_eq!(config.target_width, Some(1280));
    assert_eq!(config.target_height, Some(720));

    // Verify source type
    match config.source {
        VideoSource::Camera(ref device) => {
            assert_eq!(device, "/dev/video0");
        }
        _ => panic!("Expected Camera source"),
    }
}

#[cfg(any(feature = "video", feature = "video-pure"))]
#[test]
fn test_video_frame_conversions() {
    use kizzasi_io::VideoFrame;

    // Create RGB frame
    let rgb_frame = VideoFrame {
        index: 0,
        timestamp: 0.0,
        width: 2,
        height: 2,
        channels: 3,
        data: vec![
            255, 0, 0, // Red
            0, 255, 0, // Green
            0, 0, 255, // Blue
            255, 255, 255, // White
        ],
    };

    // Convert to grayscale (fallible now: a frame whose data length does not
    // match its dimensions returns an error instead of panicking)
    let gray_frame = rgb_frame.to_grayscale().expect("valid RGB frame");
    assert_eq!(gray_frame.channels, 1);
    assert_eq!(gray_frame.data.len(), 4);

    // Convert to normalized f32
    let normalized = rgb_frame.to_normalized_f32();
    assert_eq!(normalized.len(), 12); // 2x2x3
    assert!(normalized.iter().all(|&x| (0.0..=1.0).contains(&x)));

    // Test array conversion
    let array = rgb_frame.to_array().expect("valid RGB frame");
    assert_eq!(array.shape(), &[2, 2, 3]);
}

#[test]
fn test_signal_quality_metrics() {
    use kizzasi_io::HealthMonitor;

    // Create health monitor
    let mut monitor = HealthMonitor::new();

    // Record some clean signals
    let clean_signal = vec![0.5f32; 1000];
    monitor.record_samples(&clean_signal);

    let health = monitor.health();
    let quality = monitor.signal_quality();

    // Should have processed samples
    assert_eq!(health.samples_processed, 1000);

    // Signal quality should be calculated
    assert!(quality.snr_db.is_finite());
    assert!(quality.crest_factor > 0.0);
}

#[test]
fn test_aggregate_health_monitoring() {
    use kizzasi_io::{AggregateHealth, HealthMonitor};

    // Create multiple monitors
    let mut monitor1 = HealthMonitor::new();
    let mut monitor2 = HealthMonitor::new();

    // Record some activity
    monitor1.record_samples(&vec![0.0; 100]);
    monitor2.record_samples(&vec![0.0; 200]);

    monitor1.record_underrun();
    monitor2.record_underrun();

    // Aggregate health from both monitors
    let healths = vec![monitor1.health(), monitor2.health()];
    let aggregate = AggregateHealth::from_streams(&healths);

    // Should have combined statistics
    assert_eq!(aggregate.stream_count, 2);
    assert!(aggregate.total_samples >= 300);
    assert!(aggregate.total_underruns >= 2);
}

#[tokio::test]
async fn test_channel_stream_async() {
    use kizzasi_io::{ChannelStream, StreamConfig};
    use tokio::sync::mpsc;

    // Create channel
    let (tx, rx) = mpsc::channel(10);

    // Create stream
    let config = StreamConfig::default().buffer_size(4);
    let mut stream = ChannelStream::new(config, rx);

    // Send some data
    tx.send(vec![1.0, 2.0, 3.0, 4.0]).await.unwrap();

    // Read from stream. Per the stream read contract the block contains only
    // real samples -- this test used to assert a 1024-long buffer in which
    // 1020 samples were fabricated zeros.
    let data = stream.read().await.unwrap();

    assert_eq!(data.len(), 4);
    assert_eq!(data[0], 1.0);
    assert_eq!(data[1], 2.0);
    assert_eq!(data[2], 3.0);
    assert_eq!(data[3], 4.0);

    // Closing the producer ends the stream instead of yielding zeros forever.
    drop(tx);
    assert!(matches!(
        stream.read().await,
        Err(kizzasi_io::IoError::EndOfStream)
    ));
}

#[test]
fn test_buffer_pool_efficiency() {
    use kizzasi_io::BufferPool;

    // Create buffer pool with capacity
    let pool = BufferPool::new(10);

    // Acquire buffers
    let buf1 = pool.acquire();
    let buf2 = pool.acquire();

    // Buffers should be allocated
    assert_eq!(buf1.len(), 0); // Initially empty
    assert_eq!(buf2.len(), 0);

    // Return buffers (they go out of scope)
    drop(buf1);
    drop(buf2);

    // Buffers should be returned to pool
    // Note: Due to Arc mechanics, this might not be immediate
}
