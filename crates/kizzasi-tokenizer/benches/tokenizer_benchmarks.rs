use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kizzasi_tokenizer::*;
use std::hint::black_box;

fn continuous_tokenizer_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("continuous_tokenizer");

    for size in [16, 64, 256, 1024].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let tokenizer = ContinuousTokenizer::new(*size, 128);
        let signal = Array1::from_vec((0..*size).map(|i| (i as f32 * 0.01).sin()).collect());

        group.bench_with_input(BenchmarkId::new("encode", size), size, |b, _| {
            b.iter(|| tokenizer.encode(black_box(&signal)))
        });

        let encoded = tokenizer.encode(&signal).unwrap();
        group.bench_with_input(BenchmarkId::new("decode", size), size, |b, _| {
            b.iter(|| tokenizer.decode(black_box(&encoded)))
        });
    }

    group.finish();
}

fn quantizer_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantizers");

    let signal = Array1::from_vec((0..1024).map(|i| (i as f32 * 0.01).sin()).collect());

    // Linear quantizer
    let linear_8bit = LinearQuantizer::new(-1.0, 1.0, 8).unwrap();
    group.bench_function("linear_8bit_encode", |b| {
        b.iter(|| linear_8bit.encode(black_box(&signal)))
    });

    let linear_16bit = LinearQuantizer::new(-1.0, 1.0, 16).unwrap();
    group.bench_function("linear_16bit_encode", |b| {
        b.iter(|| linear_16bit.encode(black_box(&signal)))
    });

    // Mu-law codec
    let mulaw = MuLawCodec::new(8);
    group.bench_function("mulaw_8bit_encode", |b| {
        b.iter(|| mulaw.encode(black_box(&signal)))
    });

    let encoded = mulaw.encode(&signal).unwrap();
    group.bench_function("mulaw_8bit_decode", |b| {
        b.iter(|| mulaw.decode(black_box(&encoded)))
    });

    group.finish();
}

fn multiscale_tokenizer_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiscale_tokenizer");

    for num_levels in [2, 3, 4].iter() {
        let tokenizer = MultiScaleTokenizer::with_factors(
            256,
            32,
            &(0..*num_levels).map(|i| 1 << i).collect::<Vec<_>>(),
        );
        let signal = Array1::from_vec((0..256).map(|i| (i as f32 * 0.01).sin()).collect());

        group.bench_with_input(
            BenchmarkId::new("encode", num_levels),
            num_levels,
            |b, _| b.iter(|| tokenizer.encode(black_box(&signal))),
        );

        let encoded = tokenizer.encode(&signal).unwrap();
        group.bench_with_input(
            BenchmarkId::new("decode", num_levels),
            num_levels,
            |b, _| b.iter(|| tokenizer.decode(black_box(&encoded))),
        );
    }

    group.finish();
}

#[cfg(feature = "vqvae")]
fn vqvae_benchmarks(c: &mut Criterion) {
    use kizzasi_tokenizer::vqvae::*;

    let mut group = c.benchmark_group("vqvae");

    for codebook_size in [128, 256, 512].iter() {
        let config = VQConfig {
            codebook_size: *codebook_size,
            embed_dim: 64,
            ..Default::default()
        };

        let tokenizer = VQVAETokenizer::new(128, config);
        let signal = Array1::from_vec((0..128).map(|i| (i as f32 * 0.01).sin()).collect());

        group.bench_with_input(
            BenchmarkId::new("encode", codebook_size),
            codebook_size,
            |b, _| b.iter(|| tokenizer.encode_quantized(black_box(&signal))),
        );

        let (idx, _) = tokenizer.encode_quantized(&signal).unwrap();
        group.bench_with_input(
            BenchmarkId::new("decode", codebook_size),
            codebook_size,
            |b, _| b.iter(|| tokenizer.decode_from_index(black_box(idx))),
        );
    }

    // Vector quantizer operations
    let vq = VectorQuantizer::new(VQConfig {
        codebook_size: 512,
        embed_dim: 64,
        ..Default::default()
    });

    let vector = Array1::from_vec((0..64).map(|i| (i as f32 * 0.01).sin()).collect());

    group.bench_function("vq_find_nearest", |b| {
        b.iter(|| vq.find_nearest(black_box(&vector)))
    });

    group.bench_function("vq_quantize", |b| {
        b.iter(|| vq.quantize(black_box(&vector)))
    });

    // Batch quantization
    let vectors: Vec<Array1<f32>> = (0..100)
        .map(|i| Array1::from_vec((0..64).map(|j| ((i + j) as f32 * 0.01).sin()).collect()))
        .collect();

    group.bench_function("vq_quantize_batch", |b| {
        b.iter(|| vq.quantize_batch(black_box(&vectors)))
    });

    group.finish();
}

fn batch_processing_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");

    for batch_size in [1, 10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        let tokenizer = ContinuousTokenizer::new(64, 128);
        let signals = Array2::from_shape_fn((*batch_size, 64), |(i, j)| {
            ((i * 64 + j) as f32 * 0.01).sin()
        });

        group.bench_with_input(
            BenchmarkId::new("encode_batch", batch_size),
            batch_size,
            |b, _| b.iter(|| tokenizer.encode_batch(black_box(&signals))),
        );

        let encoded = tokenizer.encode_batch(&signals).unwrap();
        group.bench_with_input(
            BenchmarkId::new("decode_batch", batch_size),
            batch_size,
            |b, _| b.iter(|| tokenizer.decode_batch(black_box(&encoded))),
        );
    }

    group.finish();
}

fn streaming_tokenizer_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_tokenizer");

    for signal_len in [512, 1024, 2048, 4096].iter() {
        group.throughput(Throughput::Elements(*signal_len as u64));

        let tokenizer = ContinuousTokenizer::new(64, 128);
        let streaming = StreamingTokenizer::new(tokenizer, 64, 8).unwrap();
        let signal = Array1::from_vec((0..*signal_len).map(|i| (i as f32 * 0.01).sin()).collect());

        group.bench_with_input(
            BenchmarkId::new("encode_streaming", signal_len),
            signal_len,
            |b, _| b.iter(|| streaming.encode_streaming(black_box(&signal))),
        );

        let chunks = streaming.encode_streaming(&signal).unwrap();
        group.bench_with_input(
            BenchmarkId::new("decode_streaming", signal_len),
            signal_len,
            |b, _| b.iter(|| streaming.decode_streaming(black_box(&chunks))),
        );
    }

    group.finish();
}

fn pooling_methods_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("pooling_methods");

    let signal = Array1::from_vec((0..256).map(|i| (i as f32 * 0.01).sin()).collect());

    for method in [PoolMethod::Stride, PoolMethod::Average, PoolMethod::Max].iter() {
        let tokenizer = MultiScaleTokenizer::new(256, 32).with_pool_method(*method);

        group.bench_with_input(
            BenchmarkId::new("encode", format!("{:?}", method)),
            method,
            |b, _| b.iter(|| tokenizer.encode(black_box(&signal))),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    continuous_tokenizer_benchmarks,
    quantizer_benchmarks,
    multiscale_tokenizer_benchmarks,
    batch_processing_benchmarks,
    streaming_tokenizer_benchmarks,
    pooling_methods_benchmarks,
);

#[cfg(feature = "vqvae")]
criterion_group!(vqvae_benches, vqvae_benchmarks);

#[cfg(feature = "vqvae")]
criterion_main!(benches, vqvae_benches);

#[cfg(not(feature = "vqvae"))]
criterion_main!(benches);
