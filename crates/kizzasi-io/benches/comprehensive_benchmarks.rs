//! Comprehensive benchmarks for kizzasi-io
//!
//! Benchmarks performance-critical operations including:
//! - Signal processing (FFT, filtering, resampling)
//! - Advanced signal processing (HHT, adaptive filters, cepstral analysis)
//! - Time-frequency analysis (Gabor, S-transform, Wigner-Ville, Choi-Williams)
//! - Source separation (FastICA, NMF, PCA, Temporal decorrelation)
//! - Beamforming (Delay-and-Sum, MVDR, Adaptive, DOA)
//! - Quality metrics (PESQ, STOI, POLQA, MOS)
//! - Advanced resampling (Farrow, time-varying, arbitrary SRC)
//! - Optical flow computation
//! - Lock-free data structures
//! - Zero-copy buffers
//! - Audio/video processing

use criterion::{
    black_box, criterion_group, criterion_main, Bencher, BenchmarkId, Criterion, Throughput,
};
use kizzasi_io::*;
use scirs2_core::ndarray::{Array1, Array2};

// Signal Processing Benchmarks

fn bench_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");

    for size in [256, 512, 1024, 2048, 4096, 8192].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b: &mut Bencher, &size| {
                let signal = Array1::from_vec((0..size).map(|i| (i as f32).sin()).collect());
                let mut processor = SignalProcessor::new(size);

                b.iter(|| processor.fft(black_box(&signal)));
            },
        );
    }
    group.finish();
}

fn bench_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("filtering");

    let sizes = [1024, 4096, 8192];
    let signal = Array1::from_vec((0..8192).map(|i| (i as f32).sin()).collect());

    // IIR filtering
    group.bench_function("iir_butterworth_lowpass", |b: &mut Bencher| {
        let mut filter = IirFilter::butterworth_lowpass(0.1).expect("filter creation");
        b.iter(|| filter.process(black_box(&signal)));
    });

    // FIR filtering
    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("fir_lowpass", size),
            &size,
            |b: &mut Bencher, &size| {
                let sig = Array1::from_vec((0..size).map(|i| (i as f32).sin()).collect());
                let mut filter = FirFilter::sinc_lowpass(0.1, 51).expect("filter creation");

                b.iter(|| filter.process(black_box(&sig)));
            },
        );
    }

    group.finish();
}

fn bench_resampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("resampling");

    // Streaming resampler (44.1kHz -> 48kHz)
    group.bench_function("streaming_resample_44100_to_48000", |b: &mut Bencher| {
        let signal: Vec<f32> = (0..4410).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut resampler = StreamingResampler::new(44100.0, 48000.0);

        b.iter(|| resampler.process(black_box(&signal)));
    });

    // Sinc resampler (high quality)
    group.bench_function("sinc_streaming_resample", |b: &mut Bencher| {
        let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut resampler = SincStreamingResampler::new(44100.0, 48000.0, 64);

        b.iter(|| resampler.process(black_box(&signal)));
    });

    // Cubic resampler
    group.bench_function("cubic_resample_44100_to_48000", |b: &mut Bencher| {
        let signal: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
        let resampler = CubicResampler::new(44100.0, 48000.0);

        b.iter(|| resampler.resample(black_box(&signal)));
    });

    group.finish();
}

fn bench_spectrogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectrogram");

    let signal = Array1::from_vec((0..44100).map(|i| (i as f32 * 0.01).sin()).collect());
    let mut processor = SignalProcessor::new(44100);

    group.bench_function("spectrogram_2048_512", |b: &mut Bencher| {
        b.iter(|| processor.spectrogram(black_box(&signal), 2048, 512, WindowType::Hann));
    });

    group.finish();
}

fn bench_mfcc(c: &mut Criterion) {
    let mut group = c.benchmark_group("mfcc");

    let signal = Array1::from_vec((0..8192).map(|i| (i as f32 * 0.01).sin()).collect());
    let mut processor = SignalProcessor::new(8192);

    // mfcc(signal, n_mfcc, n_fft, hop_length, n_mels)
    group.bench_function("mfcc_extraction", |b: &mut Bencher| {
        b.iter(|| processor.mfcc(black_box(&signal), 13, 512, 256, 40));
    });

    group.finish();
}

fn bench_wavelet_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("wavelet");

    let signal = (0..1024)
        .map(|i| (i as f32 * 0.01).sin())
        .collect::<Vec<_>>();
    let analyzer = WaveletAnalyzer::new(WaveletType::Daubechies4);

    group.bench_function("dwt_db4", |b: &mut Bencher| {
        b.iter(|| analyzer.dwt(black_box(&signal)));
    });

    group.bench_function("dwt_multilevel_3", |b: &mut Bencher| {
        b.iter(|| analyzer.dwt_multilevel(black_box(&signal), 3));
    });

    group.finish();
}

// Optical Flow Benchmarks

#[cfg(feature = "video")]
fn bench_optical_flow(c: &mut Criterion) {
    let mut group = c.benchmark_group("optical_flow");

    // Create test frames
    let width = 640;
    let height = 480;
    let frame1_data = vec![128u8; width * height];
    let frame2_data = vec![130u8; width * height];

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

    // Dense gradient method
    group.bench_function("optical_flow_dense_gradient", |b: &mut Bencher| {
        let estimator = OpticalFlowEstimator::new(OpticalFlowMethod::DenseGradient);
        b.iter(|| estimator.compute(black_box(&frame1), black_box(&frame2)));
    });

    // Lucas-Kanade method
    group.bench_function("optical_flow_lucas_kanade", |b: &mut Bencher| {
        let estimator =
            OpticalFlowEstimator::new(OpticalFlowMethod::LucasKanade).with_window_size(15);
        b.iter(|| estimator.compute(black_box(&frame1), black_box(&frame2)));
    });

    // Block matching method
    group.bench_function("optical_flow_block_matching", |b: &mut Bencher| {
        let estimator =
            OpticalFlowEstimator::new(OpticalFlowMethod::BlockMatching).with_window_size(16);
        b.iter(|| estimator.compute(black_box(&frame1), black_box(&frame2)));
    });

    group.finish();
}

// Lock-Free Data Structures Benchmarks

fn bench_lockfree_queue(c: &mut Criterion) {
    let mut group = c.benchmark_group("lockfree");

    // Lock-free queue
    group.bench_function("lockfree_queue_push_pop", |b: &mut Bencher| {
        let queue = LockFreeQueue::new(1024);
        let mut counter = 0.0f32;

        b.iter(|| {
            queue.try_push(counter);
            counter += 1.0;
            queue.pop()
        });
    });

    // Lock-free ring buffer
    group.bench_function("lockfree_ringbuffer_write_read", |b: &mut Bencher| {
        let mut buffer = LockFreeRingBuffer::new(1024);

        b.iter(|| {
            buffer.write(black_box(1.0f32));
            buffer.read()
        });
    });

    // Signal queue (takes capacity and batch_size)
    group.bench_function("signal_queue_write_read", |b: &mut Bencher| {
        let queue = SignalQueue::new(1024, 128);
        let signal = vec![1.0f32; 128];

        b.iter(|| {
            let _ = queue.write_samples(black_box(&signal));
            queue.read_batch()
        });
    });

    group.finish();
}

// Zero-Copy Buffers Benchmarks

fn bench_zerocopy_buffers(c: &mut Criterion) {
    let mut group = c.benchmark_group("zerocopy");

    let data = vec![1.0f32; 8192];

    // Shared signal buffer
    group.bench_function("shared_buffer_create_slice", |b: &mut Bencher| {
        b.iter(|| {
            let buffer = SharedSignalBuffer::new(black_box(data.clone()));
            buffer.slice(0, 4096)
        });
    });

    // Buffer pool
    group.bench_function("buffer_pool_acquire_release", |b: &mut Bencher| {
        let pool = BufferPool::new(10);

        b.iter(|| {
            let buf = pool.acquire();
            drop(buf);
        });
    });

    group.finish();
}

// Ring Buffer Benchmarks

fn bench_ring_buffers(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer");

    // Standard ring buffer
    group.bench_function("ring_buffer_push_pop", |b: &mut Bencher| {
        let mut buffer = RingBuffer::new(1024);
        let mut counter = 0.0f32;

        b.iter(|| {
            buffer.push(counter);
            counter += 1.0;
            buffer.pop()
        });
    });

    // Signal ring buffer with statistics
    group.bench_function("signal_ring_buffer_with_stats", |b: &mut Bencher| {
        let mut buffer = SignalRingBuffer::new(1024);
        let samples = vec![1.0f32; 128];

        b.iter(|| {
            buffer.push_slice(black_box(&samples));
            buffer.mean();
            buffer.rms()
        });
    });

    group.finish();
}

// Health Monitoring Benchmarks

fn bench_health_monitoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("health_monitoring");

    let samples = vec![0.5f32; 1024];

    group.bench_function("health_monitor_record", |b: &mut Bencher| {
        let mut monitor = HealthMonitor::new();

        b.iter(|| {
            monitor.record_samples(black_box(&samples));
        });
    });

    group.bench_function("signal_quality_analysis", |b: &mut Bencher| {
        let mut monitor = HealthMonitor::new();
        monitor.record_samples(&samples);

        b.iter(|| monitor.signal_quality());
    });

    group.finish();
}

// Signal Generator Benchmarks

fn bench_signal_generators(c: &mut Criterion) {
    let mut group = c.benchmark_group("generators");

    let n_samples = 8192;

    group.bench_function("sine_generator", |b: &mut Bencher| {
        let mut gen = SineGenerator::new(1000.0, 44100.0, 0.5);
        b.iter(|| gen.generate(black_box(n_samples)));
    });

    group.bench_function("chirp_generator", |b: &mut Bencher| {
        let mut gen = ChirpGenerator::new(100.0, 2000.0, 44100.0, 1.0, 0.7);
        b.iter(|| gen.generate(black_box(n_samples)));
    });

    group.bench_function("white_noise_generator", |b: &mut Bencher| {
        let mut gen = WhiteNoiseGenerator::new(0.5);
        b.iter(|| gen.generate(black_box(n_samples)));
    });

    group.finish();
}

// Recorder Benchmarks

fn bench_recorder(c: &mut Criterion) {
    let mut group = c.benchmark_group("recorder");

    let signal = Array1::from_vec((0..1024).map(|i| (i as f32).sin()).collect());

    // Note: StreamRecorder::new is async, so we benchmark the synchronous parts
    group.bench_function("signal_creation", |b: &mut Bencher| {
        b.iter(|| Array1::from_vec((0..1024).map(|i| (i as f32).sin()).collect::<Vec<f32>>()));
    });

    group.bench_function("config_creation", |b: &mut Bencher| {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("bench_recording.bin");

        b.iter(|| RecorderConfig {
            path: path.to_str().expect("path").to_string(),
            format: RecorderFormat::Binary,
            sample_rate: 44100.0,
            channels: 1,
            buffer_size: 1024,
            metadata: std::collections::HashMap::new(),
            record_timestamps: false,
        });
    });

    let _ = signal; // Suppress unused warning

    group.finish();
}

// Advanced Signal Processing Benchmarks (Phase 3-5)

fn bench_hht(c: &mut Criterion) {
    let mut group = c.benchmark_group("hht");

    // Create test signal: multi-component signal
    let sample_rate = 1000.0;
    let duration = 1.0;
    let n = (sample_rate * duration) as usize;
    let mut signal = vec![0.0; n];
    use std::f64::consts::PI;

    for (i, sig) in signal.iter_mut().enumerate() {
        let t = i as f64 / sample_rate;
        *sig = (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 20.0 * t).sin();
    }

    // EMD benchmark
    group.bench_function("emd_decomposition", |b: &mut Bencher| {
        let config = EmdConfig::default();
        let emd = EmpiricalModeDecomposition::new(sample_rate, config);

        b.iter(|| emd.decompose(black_box(&signal)));
    });

    // EEMD benchmark (smaller ensemble for speed)
    group.bench_function("eemd_ensemble_5", |b: &mut Bencher| {
        let config = EmdConfig::default();
        let eemd = EnsembleEmd::new(sample_rate, config, 5, 0.1);

        b.iter(|| eemd.decompose(black_box(&signal)));
    });

    group.finish();
}

fn bench_quality_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_metrics");

    // Create test signals
    let sample_rate = 8000.0;
    let n = 8000; // 1 second
    let reference: Vec<f64> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sample_rate).sin())
        .collect();
    let degraded: Vec<f64> = reference.iter().map(|&x| x * 0.9 + 0.01).collect();

    // SNR calculation
    group.bench_function("snr_calculation", |b: &mut Bencher| {
        let snr_calc = SnrCalculator::new(256);
        b.iter(|| snr_calc.calculate_snr(black_box(&reference), black_box(&degraded)));
    });

    // Segmental SNR
    group.bench_function("segmental_snr", |b: &mut Bencher| {
        let snr_calc = SnrCalculator::new(256);
        b.iter(|| snr_calc.calculate_segmental_snr(black_box(&reference), black_box(&degraded)));
    });

    // STOI calculation
    group.bench_function("stoi_calculation", |b: &mut Bencher| {
        let stoi_calc = StoiCalculator::new(sample_rate);
        b.iter(|| stoi_calc.calculate(black_box(&reference), black_box(&degraded)));
    });

    // PESQ calculation
    group.bench_function("pesq_calculation", |b: &mut Bencher| {
        let pesq_calc = PesqCalculator::new(sample_rate);
        b.iter(|| pesq_calc.calculate(black_box(&reference), black_box(&degraded)));
    });

    // POLQA calculation
    group.bench_function("polqa_calculation", |b: &mut Bencher| {
        let polqa_calc = PolqaCalculator::new(sample_rate);
        b.iter(|| polqa_calc.calculate(black_box(&reference), black_box(&degraded)));
    });

    // MOS prediction
    group.bench_function("mos_prediction", |b: &mut Bencher| {
        let mos_predictor = MosPredictor::new(sample_rate);
        b.iter(|| mos_predictor.predict(black_box(&reference), black_box(&degraded)));
    });

    group.finish();
}

fn bench_advanced_resampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_resampling");

    let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();

    // Farrow resampler
    group.bench_function("farrow_cubic_resampler", |b: &mut Bencher| {
        let mut resampler = FarrowResampler::new_cubic(44100.0, 48000.0);
        b.iter(|| resampler.process(black_box(&signal)));
    });

    // Time-varying resampler
    group.bench_function("time_varying_resampler_sinusoidal", |b: &mut Bencher| {
        let mut resampler =
            TimeVaryingResampler::new(44100.0, 48000.0, RatioModulation::Sinusoidal, 0.1, 10.0);
        b.iter(|| resampler.process(black_box(&signal)));
    });

    // Arbitrary SRC
    group.bench_function("arbitrary_src_resampler", |b: &mut Bencher| {
        let mut resampler = ArbitrarySrcResampler::new(44100.0, 48000.0);
        b.iter(|| resampler.process(black_box(&signal)));
    });

    group.finish();
}

fn bench_adaptive_filters(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_filters");

    // LMS filter
    group.bench_function("lms_filter", |b: &mut Bencher| {
        let input = vec![1.0f32; 100];
        let desired = vec![0.9f32; 100];
        if let Ok(mut filter) = LmsFilter::new(32, 0.01) {
            b.iter(|| {
                for i in 0..100 {
                    let _ = filter.adapt(black_box(input[i]), black_box(desired[i]));
                }
            });
        }
    });

    // NLMS filter
    group.bench_function("nlms_filter", |b: &mut Bencher| {
        let input = vec![1.0f32; 100];
        let desired = vec![0.9f32; 100];
        if let Ok(mut filter) = NlmsFilter::new(32, 0.5, None) {
            b.iter(|| {
                for i in 0..100 {
                    let _ = filter.adapt(black_box(input[i]), black_box(desired[i]));
                }
            });
        }
    });

    // RLS filter
    group.bench_function("rls_filter", |b: &mut Bencher| {
        let input = vec![1.0f32; 100];
        let desired = vec![0.9f32; 100];
        if let Ok(mut filter) = RlsFilter::new(32, 0.99, 0.01) {
            b.iter(|| {
                for i in 0..100 {
                    let _ = filter.adapt(black_box(input[i]), black_box(desired[i]));
                }
            });
        }
    });

    // Particle filter benchmark skipped - requires complex initialization
    // ParticleFilter::new takes (num_particles, initial_state, initial_std, process_noise_std, measurement_noise_std)

    group.finish();
}

fn bench_cepstral_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("cepstral");

    let signal_f32 = Array1::from_vec((0..1024).map(|i| (i as f32 * 0.01).sin()).collect());

    // Real cepstrum
    group.bench_function("real_cepstrum", |b: &mut Bencher| {
        let mut cepstral = RealCepstrum::new();
        b.iter(|| cepstral.compute(black_box(&signal_f32)));
    });

    // Complex cepstrum
    group.bench_function("complex_cepstrum", |b: &mut Bencher| {
        let mut cepstral = ComplexCepstrum::new();
        b.iter(|| cepstral.compute(black_box(&signal_f32)));
    });

    // Formant tracking
    group.bench_function("formant_tracking", |b: &mut Bencher| {
        let mut tracker = FormantTracker::new(16000.0, None);
        b.iter(|| tracker.estimate_formants(black_box(&signal_f32), 4));
    });

    group.finish();
}

fn bench_timefreq_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("timefreq");

    let signal_f32 = Array1::from_vec((0..1024).map(|i| (i as f32 * 0.01).sin()).collect());
    let sample_rate = 1000.0_f32;

    // Gabor transform
    group.bench_function("gabor_transform", |b: &mut Bencher| {
        let mut gabor = GaborTransform::new(sample_rate);
        b.iter(|| gabor.compute(black_box(&signal_f32), 32, 8, 8.0));
    });

    // S-transform
    group.bench_function("s_transform", |b: &mut Bencher| {
        let mut stransform = STransform::new(sample_rate);
        b.iter(|| stransform.compute(black_box(&signal_f32), 1.0));
    });

    // Wigner-Ville distribution
    group.bench_function("wigner_ville", |b: &mut Bencher| {
        let mut wv = WignerVille::new(sample_rate);
        b.iter(|| wv.compute(black_box(&signal_f32)));
    });

    // Choi-Williams distribution
    group.bench_function("choi_williams", |b: &mut Bencher| {
        let mut cw = ChoiWilliams::new(sample_rate, 1.0);
        b.iter(|| cw.compute(black_box(&signal_f32)));
    });

    // Reassigned spectrogram
    group.bench_function("reassigned_spectrogram", |b: &mut Bencher| {
        let mut rs = ReassignedSpectrogram::new(sample_rate);
        b.iter(|| rs.compute(black_box(&signal_f32), 64, 16, WindowType::Hann));
    });

    group.finish();
}

fn bench_source_separation(c: &mut Criterion) {
    let mut group = c.benchmark_group("source_separation");

    // Create mixed signals for benchmarking
    let n_samples = 1000;
    let n_channels = 3;
    let mut mixed_signals = Array2::zeros((n_channels, n_samples));
    for i in 0..n_channels {
        for j in 0..n_samples {
            mixed_signals[[i, j]] = ((i + j) as f32 * 0.01).sin();
        }
    }

    // FastICA (takes n_components, max_iter, tolerance)
    group.bench_function("fastica_3ch_1000samples", |b: &mut Bencher| {
        let fastica = FastICA::new(n_channels, Some(100), Some(1e-4));
        b.iter(|| fastica.fit_transform(black_box(&mixed_signals)));
    });

    // NMF (requires non-negative data)
    group.bench_function("nmf_3ch_1000samples", |b: &mut Bencher| {
        let nonneg_signals = &mixed_signals + 1.0; // Make non-negative
        let nmf = NMF::new(n_channels, Some(50), Some(1e-3));
        b.iter(|| nmf.fit_transform(black_box(&nonneg_signals)));
    });

    // PCA
    group.bench_function("pca_3ch_1000samples", |b: &mut Bencher| {
        let pca = PCA::new(n_channels);
        b.iter(|| pca.fit_transform(black_box(&mixed_signals)));
    });

    // Temporal decorrelation
    group.bench_function("temporal_decorrelation", |b: &mut Bencher| {
        let temporal = TemporalDecorrelation::new(10);
        b.iter(|| temporal.separate(black_box(&mixed_signals)));
    });

    group.finish();
}

fn bench_beamforming(c: &mut Criterion) {
    let mut group = c.benchmark_group("beamforming");

    // Create synthetic microphone array signals
    let sample_rate = 16000.0;
    let num_mics = 4;
    let n_samples = 1600; // 0.1 seconds

    let mic_signals_f32: Vec<Vec<f32>> = (0..num_mics)
        .map(|mic_idx| {
            (0..n_samples)
                .map(|i| {
                    let t = i as f32 / sample_rate;
                    let delay = mic_idx as f32 * 0.0001; // Simulated propagation delay
                    (2.0 * std::f32::consts::PI * 1000.0 * (t - delay)).sin()
                })
                .collect()
        })
        .collect();

    // Convert to Array2 for beamforming
    let mut signals_array = Array2::zeros((num_mics, n_samples));
    for (i, sig) in mic_signals_f32.iter().enumerate() {
        for (j, &val) in sig.iter().enumerate() {
            signals_array[[i, j]] = val;
        }
    }

    // Transpose to (n_samples, n_mics) for beamforming API
    let signals_transposed = signals_array.t().to_owned();

    // Delay-and-Sum beamforming
    group.bench_function("delay_and_sum", |b: &mut Bencher| {
        let mic_array = MicrophoneArray::linear(num_mics, 0.05, sample_rate);
        let das = DelayAndSum::new(mic_array);
        let azimuth = 0.0_f32;
        let elevation = 0.0_f32;

        b.iter(|| das.process(black_box(&signals_transposed), azimuth, elevation));
    });

    // MVDR beamforming
    group.bench_function("mvdr_beamforming", |b: &mut Bencher| {
        let mic_array = MicrophoneArray::linear(num_mics, 0.05, sample_rate);
        let mvdr = MVDR::new(mic_array, Some(1e-3));
        let azimuth = 0.0_f32;
        let elevation = 0.0_f32;

        b.iter(|| mvdr.process(black_box(&signals_transposed), azimuth, elevation));
    });

    // Adaptive beamforming (processes one sample at a time)
    group.bench_function("adaptive_beamformer", |b: &mut Bencher| {
        let mic_array = MicrophoneArray::linear(num_mics, 0.05, sample_rate);
        let mut adaptive = AdaptiveBeamformer::new(mic_array, 0.01);
        // Extract first sample from all mics
        let input_sample = signals_transposed.row(0).to_owned();
        let desired = 1.0_f32;

        b.iter(|| adaptive.adapt(black_box(&input_sample), black_box(desired)));
    });

    // DOA estimation (reduced angular resolution for speed)
    group.bench_function("doa_estimation", |b: &mut Bencher| {
        let mic_array = MicrophoneArray::linear(num_mics, 0.05, sample_rate);
        let doa = DOAEstimator::new(mic_array);
        let azimuth_resolution: usize = 19;

        b.iter(|| {
            doa.estimate_srp(
                black_box(&signals_transposed),
                black_box(azimuth_resolution),
            )
        });
    });

    group.finish();
}

// Define benchmark groups
criterion_group!(
    signal_processing,
    bench_fft,
    bench_filtering,
    bench_resampling,
    bench_spectrogram,
    bench_mfcc,
    bench_wavelet_transform,
);

criterion_group!(
    advanced_signal_processing,
    bench_hht,
    bench_quality_metrics,
    bench_advanced_resampling,
    bench_adaptive_filters,
    bench_cepstral_analysis,
    bench_timefreq_analysis,
    bench_source_separation,
    bench_beamforming,
);

#[cfg(feature = "video")]
criterion_group!(video_processing, bench_optical_flow,);

criterion_group!(
    data_structures,
    bench_lockfree_queue,
    bench_zerocopy_buffers,
    bench_ring_buffers,
);

criterion_group!(monitoring, bench_health_monitoring,);

criterion_group!(generators, bench_signal_generators, bench_recorder,);

// Main benchmark runner
#[cfg(feature = "video")]
criterion_main!(
    signal_processing,
    advanced_signal_processing,
    video_processing,
    data_structures,
    monitoring,
    generators,
);

#[cfg(not(feature = "video"))]
criterion_main!(
    signal_processing,
    advanced_signal_processing,
    data_structures,
    monitoring,
    generators,
);
