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

## Supported I/O

- Audio: CPAL (cross-platform), JACK (Linux), ASIO (Windows)
- Network: MQTT, WebSocket, TCP/UDP, Serial, OSC, ZeroMQ
- Video: FFmpeg, V4L2, DirectShow, AVFoundation
- Files: WAV, CSV, HDF5
- Optional: ROS2 bridge (requires `ros2` feature)

## Documentation

- [API Documentation](https://docs.rs/kizzasi-io)
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
