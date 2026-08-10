//! Comprehensive benchmarks for kizzasi-tokenizer
//!
//! This benchmark suite covers:
//! - Advanced quantization methods
//! - Specialized tokenizers
//! - Advanced features (dropout, jitter, temporal coherence, hierarchical)
//! - Entropy coding
//! - SIMD and GPU acceleration
//! - Memory and throughput analysis

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kizzasi_tokenizer::*;
use std::hint::black_box;

// ============================================================================
// Advanced Quantization Benchmarks
// ============================================================================

fn advanced_quantization_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_quantization");

    let signal = Array1::from_vec((0..1024).map(|i| (i as f32 * 0.01).sin()).collect());

    // Adaptive quantizer
    let adaptive = AdaptiveQuantizer::new(8, 32, 0.5, -1.0, 1.0).unwrap();
    group.bench_function("adaptive_quant_encode", |b| {
        b.iter(|| adaptive.encode(black_box(&signal)))
    });

    let encoded_adaptive = adaptive.encode(&signal).unwrap();
    group.bench_function("adaptive_quant_decode", |b| {
        b.iter(|| adaptive.decode(black_box(&encoded_adaptive)))
    });

    // Dead zone quantizer
    let deadzone = DeadZoneQuantizer::new(8, 0.1, -1.0, 1.0).unwrap();
    group.bench_function("deadzone_quant_encode", |b| {
        b.iter(|| deadzone.encode(black_box(&signal)))
    });

    // Non-uniform quantizer
    let bin_edges: Vec<f32> = (0..=256).map(|i| -1.0 + 2.0 * i as f32 / 256.0).collect();
    let recon_values: Vec<f32> = (0..256)
        .map(|i| -1.0 + 2.0 * (i as f32 + 0.5) / 256.0)
        .collect();
    let nonuniform = NonUniformQuantizer::new(bin_edges, recon_values).unwrap();

    group.bench_function("nonuniform_quant_encode", |b| {
        b.iter(|| nonuniform.encode(black_box(&signal)))
    });

    group.finish();
}

// ============================================================================
// Specialized Tokenizers Benchmarks
// ============================================================================

fn specialized_tokenizers_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("specialized_tokenizers");

    // Ensure signal length is power of 2 for wavelets
    let signal = Array1::from_vec((0..1024).map(|i| (i as f32 * 0.01).sin()).collect());

    // Wavelet tokenizer
    let wavelet_config = WaveletConfig {
        levels: 3,
        family: WaveletFamily::Haar,
        bits: 8,
    };
    let wavelet = WaveletTokenizer::new(wavelet_config).unwrap();

    group.bench_function("wavelet_haar_encode", |b| {
        b.iter(|| wavelet.encode(black_box(&signal)))
    });

    let encoded_wavelet = wavelet.encode(&signal).unwrap();
    group.bench_function("wavelet_haar_decode", |b| {
        b.iter(|| wavelet.decode(black_box(&encoded_wavelet)))
    });

    // Daubechies-4 wavelet
    let db4_config = WaveletConfig {
        levels: 3,
        family: WaveletFamily::Daubechies4,
        bits: 8,
    };
    let db4 = WaveletTokenizer::new(db4_config).unwrap();

    group.bench_function("wavelet_db4_encode", |b| {
        b.iter(|| db4.encode(black_box(&signal)))
    });

    // Fourier tokenizer
    let fourier_config = FourierConfig {
        num_bins: 512,
        magnitude_only: false,
        bits: 8,
    };
    let fourier = FourierTokenizer::new(fourier_config).unwrap();

    group.bench_function("fourier_full_encode", |b| {
        b.iter(|| fourier.encode(black_box(&signal)))
    });

    let encoded_fourier = fourier.encode(&signal).unwrap();
    group.bench_function("fourier_full_decode", |b| {
        b.iter(|| fourier.decode(black_box(&encoded_fourier)))
    });

    // Fourier magnitude-only
    let fourier_mag_config = FourierConfig {
        num_bins: 512,
        magnitude_only: true,
        bits: 8,
    };
    let fourier_mag = FourierTokenizer::new(fourier_mag_config).unwrap();

    group.bench_function("fourier_magnitude_encode", |b| {
        b.iter(|| fourier_mag.encode(black_box(&signal)))
    });

    // DCT tokenizer
    let dct_config = DCTConfig {
        num_coeffs: 128,
        bits: 8,
    };
    let dct = DCTTokenizer::new(dct_config).unwrap();

    group.bench_function("dct_encode", |b| b.iter(|| dct.encode(black_box(&signal))));

    let encoded_dct = dct.encode(&signal).unwrap();
    group.bench_function("dct_decode", |b| {
        b.iter(|| dct.decode(black_box(&encoded_dct)))
    });

    // K-means tokenizer
    let kmeans_config = KMeansConfig {
        num_clusters: 256,
        embed_dim: 64,
        max_iterations: 10,
        tolerance: 1e-4,
    };
    let kmeans = KMeansTokenizer::new(kmeans_config).unwrap();

    group.bench_function("kmeans_encode", |b| {
        b.iter(|| kmeans.encode(black_box(&signal)))
    });

    group.finish();
}

// ============================================================================
// Advanced Features Benchmarks
// ============================================================================

fn advanced_features_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_features");

    let tokens = Array1::from_vec((0..1024).map(|i| i as f32).collect());
    let signal = Array1::from_vec((0..1024).map(|i| (i as f32 * 0.01).sin()).collect());

    // Token dropout
    let dropout_config = TokenDropoutConfig {
        dropout_rate: 0.1,
        fill_value: 0.0,
        scale_remaining: true,
    };

    group.bench_function("token_dropout_apply", |b| {
        b.iter(|| apply_token_dropout(black_box(&tokens), &dropout_config, true))
    });

    // Batch dropout
    let batch_tokens = Array2::from_shape_fn((100, 64), |(i, j)| (i * 64 + j) as f32);
    group.bench_function("batch_token_dropout", |b| {
        b.iter(|| apply_batch_token_dropout(black_box(&batch_tokens), &dropout_config, true))
    });

    // Jitter injection
    let jitter_config = JitterConfig {
        noise_std: 0.01,
        apply_at_inference: false,
        target_snr_db: None,
    };

    group.bench_function("jitter_injection", |b| {
        b.iter(|| add_jitter(black_box(&signal), &jitter_config, true))
    });

    // Jitter with SNR target
    let jitter_snr_config = JitterConfig::with_snr(20.0);

    group.bench_function("jitter_snr_based", |b| {
        b.iter(|| add_jitter(black_box(&signal), &jitter_snr_config, true))
    });

    // Temporal coherence - EMA
    let coherence_ema = TemporalCoherenceConfig {
        smoothness: 0.5,
        window_size: 5,
        filter_type: TemporalFilterType::ExponentialMovingAverage,
    };

    group.bench_function("temporal_coherence_ema", |b| {
        b.iter(|| apply_temporal_coherence(black_box(&signal), &coherence_ema))
    });

    // Temporal coherence - SMA
    let coherence_sma = TemporalCoherenceConfig {
        smoothness: 0.5,
        window_size: 5,
        filter_type: TemporalFilterType::SimpleMovingAverage,
    };

    group.bench_function("temporal_coherence_sma", |b| {
        b.iter(|| apply_temporal_coherence(black_box(&signal), &coherence_sma))
    });

    // Temporal coherence - Gaussian
    let coherence_gauss = TemporalCoherenceConfig {
        smoothness: 0.5,
        window_size: 5,
        filter_type: TemporalFilterType::GaussianWeighted,
    };

    group.bench_function("temporal_coherence_gaussian", |b| {
        b.iter(|| apply_temporal_coherence(black_box(&signal), &coherence_gauss))
    });

    // Hierarchical tokenization
    let hier_config = HierarchicalConfig::exponential(256, 3, 0.5);
    let hierarchical = HierarchicalTokenizer::new(64, hier_config).unwrap();
    let hier_signal = Array1::from_vec((0..64).map(|i| (i as f32 * 0.01).sin()).collect());

    group.bench_function("hierarchical_encode_1_level", |b| {
        b.iter(|| hierarchical.encode_with_levels(black_box(&hier_signal), 1))
    });

    group.bench_function("hierarchical_encode_2_levels", |b| {
        b.iter(|| hierarchical.encode_with_levels(black_box(&hier_signal), 2))
    });

    group.bench_function("hierarchical_encode_3_levels", |b| {
        b.iter(|| hierarchical.encode_with_levels(black_box(&hier_signal), 3))
    });

    let indices = hierarchical.encode_with_levels(&hier_signal, 3).unwrap();
    group.bench_function("hierarchical_decode", |b| {
        b.iter(|| hierarchical.decode_hierarchical(black_box(&indices)))
    });

    group.finish();
}

// ============================================================================
// Entropy Coding Benchmarks
// ============================================================================

fn entropy_coding_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_coding");

    let symbols: Vec<u32> = (0..1000).map(|i| (i % 256) as u32).collect();

    // Compute frequency table
    group.bench_function("compute_frequencies", |b| {
        b.iter(|| compute_frequencies(black_box(&symbols)))
    });

    let frequencies = compute_frequencies(&symbols);

    // Huffman encoding
    let huffman_encoder = HuffmanEncoder::from_frequencies(&frequencies).unwrap();
    group.bench_function("huffman_build_encoder", |b| {
        b.iter(|| HuffmanEncoder::from_frequencies(black_box(&frequencies)))
    });

    group.bench_function("huffman_encode", |b| {
        b.iter(|| huffman_encoder.encode(black_box(&symbols)))
    });

    let encoded_huffman = huffman_encoder.encode(&symbols).unwrap();
    let huffman_decoder = HuffmanDecoder::new(huffman_encoder.tree());

    group.bench_function("huffman_decode", |b| {
        b.iter(|| huffman_decoder.decode(black_box(&encoded_huffman)))
    });

    // Arithmetic coding
    let mut arith_encoder = ArithmeticEncoder::from_frequencies(frequencies.clone());
    group.bench_function("arithmetic_encode", |b| {
        b.iter(|| {
            let mut encoder = ArithmeticEncoder::from_frequencies(frequencies.clone());
            encoder.encode(black_box(&symbols), false)
        })
    });

    let encoded_arith = arith_encoder.encode(&symbols, false).unwrap();
    let arith_decoder = ArithmeticDecoder::new(frequencies.clone());

    group.bench_function("arithmetic_decode", |b| {
        b.iter(|| arith_decoder.decode(black_box(&encoded_arith)))
    });

    // Range coding
    let range_encoder = RangeEncoder::from_frequencies(frequencies.clone()).unwrap();
    group.bench_function("range_encode", |b| {
        b.iter(|| range_encoder.encode(black_box(&symbols)))
    });

    group.finish();
}

// ============================================================================
// SIMD Optimization Benchmarks
// ============================================================================

#[cfg(target_arch = "x86_64")]
fn simd_benchmarks(c: &mut Criterion) {
    use kizzasi_tokenizer::simd_quant::{simd_adaptive_quantize, simd_mulaw_encode, simd_quantize};

    let mut group = c.benchmark_group("simd_optimization");

    let signal_vec: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
    let signal = Array1::from_vec(signal_vec.clone());

    // SIMD linear quantization (free function over &[f32])
    group.bench_function("simd_linear_quant_encode", |b| {
        b.iter(|| simd_quantize(black_box(&signal_vec), -1.0, 1.0, 8))
    });

    // Compare with scalar version
    let scalar_linear = LinearQuantizer::new(-1.0, 1.0, 8).unwrap();
    group.bench_function("scalar_linear_quant_encode", |b| {
        b.iter(|| scalar_linear.encode(black_box(&signal)))
    });

    // SIMD μ-law encode (free function)
    group.bench_function("simd_mulaw_encode", |b| {
        b.iter(|| simd_mulaw_encode(black_box(&signal_vec), 255.0))
    });

    // Scalar μ-law for comparison
    let scalar_mulaw = MuLawCodec::new(8);
    group.bench_function("scalar_mulaw_encode", |b| {
        b.iter(|| scalar_mulaw.encode(black_box(&signal)))
    });

    // SIMD adaptive quantize (signal, base_step, window_size, adaptation_strength)
    group.bench_function("simd_adaptive_quant", |b| {
        b.iter(|| simd_adaptive_quantize(black_box(&signal_vec), 0.01, 32, 0.5))
    });

    group.finish();
}

// ============================================================================
// GPU Acceleration Benchmarks (if available)
// ============================================================================

fn gpu_benchmarks(c: &mut Criterion) {
    use kizzasi_tokenizer::gpu_quant::*;

    let mut group = c.benchmark_group("gpu_acceleration");

    // Batch processing comparison (GPU quantizers work on batches)
    // Create flat batch of scalars for GPU quantizer
    let batch_signals_flat: Vec<f32> = (0..10000).map(|i| ((i as f32) * 0.01).sin()).collect();

    let gpu_linear = GpuLinearQuantizer::new(-1.0, 1.0, 8).unwrap();

    group.bench_function("gpu_linear_quant_batch", |b| {
        b.iter(|| gpu_linear.quantize_batch(black_box(&batch_signals_flat)))
    });

    // Compare with CPU version
    let cpu_linear = LinearQuantizer::new(-1.0, 1.0, 8).unwrap();
    let batch_signal = Array1::from_vec(batch_signals_flat.clone());

    group.bench_function("cpu_linear_quant_batch", |b| {
        b.iter(|| cpu_linear.encode(black_box(&batch_signal)))
    });

    // GPU Vector Quantizer
    #[cfg(feature = "vqvae")]
    {
        let gpu_vq = GpuVectorQuantizer::new(512, 64).unwrap();

        let batch_vecs: Vec<f32> = (0..100 * 64).map(|i| ((i as f32) * 0.01).sin()).collect();

        group.bench_function("gpu_vq_batch", |b| {
            b.iter(|| gpu_vq.quantize_vectors(black_box(&batch_vecs), 100))
        });
    }

    group.finish();
}

// ============================================================================
// Memory and Throughput Benchmarks
// ============================================================================

fn throughput_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    // Vary signal sizes for throughput measurement
    for size in [64, 256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let signal = Array1::from_vec((0..*size).map(|i| (i as f32 * 0.01).sin()).collect());

        // Linear quantizer throughput
        let quantizer = LinearQuantizer::new(-1.0, 1.0, 8).unwrap();
        group.bench_with_input(BenchmarkId::new("linear_quant", size), size, |b, _| {
            b.iter(|| quantizer.encode(black_box(&signal)))
        });

        // Continuous tokenizer throughput
        let tokenizer = ContinuousTokenizer::new(*size, 128);
        group.bench_with_input(BenchmarkId::new("continuous", size), size, |b, _| {
            b.iter(|| tokenizer.encode(black_box(&signal)))
        });
    }

    group.finish();
}

fn latency_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency");
    group.sample_size(1000); // More samples for latency measurement

    let signal = Array1::from_vec((0..256).map(|i| (i as f32 * 0.01).sin()).collect());

    // Measure end-to-end latency
    let quantizer = LinearQuantizer::new(-1.0, 1.0, 8).unwrap();

    group.bench_function("linear_roundtrip", |b| {
        b.iter(|| {
            let encoded = quantizer.encode(black_box(&signal)).unwrap();
            quantizer.decode(&encoded).unwrap()
        })
    });

    let tokenizer = ContinuousTokenizer::new(256, 128);

    group.bench_function("continuous_roundtrip", |b| {
        b.iter(|| {
            let encoded = tokenizer.encode(black_box(&signal)).unwrap();
            tokenizer.decode(&encoded).unwrap()
        })
    });

    #[cfg(feature = "vqvae")]
    {
        let vqvae = VQVAETokenizer::new(
            256,
            VQConfig {
                codebook_size: 256,
                embed_dim: 64,
                ..Default::default()
            },
        );

        group.bench_function("vqvae_roundtrip", |b| {
            b.iter(|| {
                let encoded = vqvae.encode(black_box(&signal)).unwrap();
                vqvae.decode(&encoded).unwrap()
            })
        });
    }

    group.finish();
}

// ============================================================================
// Quality Metrics Benchmarks
// ============================================================================

fn metrics_benchmarks(c: &mut Criterion) {
    use kizzasi_tokenizer::metrics::*;

    let mut group = c.benchmark_group("quality_metrics");

    let original = Array1::from_vec((0..1024).map(|i| (i as f32 * 0.01).sin()).collect());
    let reconstructed = Array1::from_vec(
        (0..1024)
            .map(|i| (i as f32 * 0.01).sin() + 0.01 * ((i as f32 * 0.1).cos()))
            .collect(),
    );

    // Quality metrics computation
    group.bench_function("compute_all_quality_metrics", |b| {
        b.iter(|| QualityMetrics::compute(black_box(&original), black_box(&reconstructed)))
    });

    group.bench_function("compute_snr", |b| {
        b.iter(|| {
            QualityMetrics::compute(black_box(&original), black_box(&reconstructed))
                .unwrap()
                .snr_db
        })
    });

    // Spectral metrics
    group.bench_function("compute_spectral_metrics", |b| {
        b.iter(|| SpectralMetrics::compute(black_box(&original), black_box(&reconstructed)))
    });

    // Compression metrics
    let compressed_size = 512;
    group.bench_function("compute_compression_metrics", |b| {
        b.iter(|| {
            CompressionMetrics::compute(
                black_box(original.len()),
                black_box(32), // 32 bits per sample (f32)
                black_box(compressed_size),
            )
        })
    });

    group.finish();
}

// ============================================================================
// Benchmark Groups
// ============================================================================

criterion_group!(
    basic_benches,
    advanced_quantization_benchmarks,
    specialized_tokenizers_benchmarks,
    advanced_features_benchmarks,
    entropy_coding_benchmarks,
);

#[cfg(target_arch = "x86_64")]
criterion_group!(simd_benches, simd_benchmarks);

criterion_group!(gpu_benches, gpu_benchmarks,);

criterion_group!(
    performance_benches,
    throughput_benchmarks,
    latency_benchmarks,
    metrics_benchmarks,
);

#[cfg(target_arch = "x86_64")]
criterion_main!(
    basic_benches,
    simd_benches,
    gpu_benches,
    performance_benches
);

#[cfg(not(target_arch = "x86_64"))]
criterion_main!(basic_benches, gpu_benches, performance_benches);
