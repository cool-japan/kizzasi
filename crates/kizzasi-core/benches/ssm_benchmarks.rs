use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kizzasi_core::*;
use scirs2_core::ndarray::{Array1, Array2};

/// Benchmark SSM forward pass at different dimensions
fn bench_ssm_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssm_forward");

    for d in [64, 128, 256].iter() {
        let config = S4DConfig {
            input_dim: *d,
            state_dim: 64,
            hidden_dim: *d,
            delta: 0.001,
            use_hippo: true,
            bidirectional: false,
        };
        let layer = S4DLayer::new(config).unwrap();
        let input = Array2::from_elem((16, *d), 0.5f32);

        group.throughput(Throughput::Elements((16 * d) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(d), d, |b, _| {
            b.iter(|| {
                black_box(layer.forward_sequence(black_box(&input)).unwrap());
            });
        });
    }
    group.finish();
}

/// Benchmark Mamba-2 forward pass
fn bench_mamba2_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("mamba2_forward");

    for d in [64, 128, 256].iter() {
        let config = Mamba2Config::new(*d, 16);
        let mut layer = Mamba2Layer::new(config).unwrap();
        let input = Array1::from_elem(*d, 0.5f32);

        group.throughput(Throughput::Elements(*d as u64));
        group.bench_with_input(BenchmarkId::from_parameter(d), d, |b, _| {
            b.iter(|| {
                black_box(layer.forward(black_box(&input)).unwrap());
            });
        });
    }
    group.finish();
}

/// Benchmark S5 forward pass
fn bench_s5_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("s5_forward");

    for d in [64, 128, 256].iter() {
        let config = S5Config::new(*d, *d, 32);
        let mut layer = S5Layer::new(config).unwrap();
        let input = Array1::from_elem(*d, 0.5f32);

        group.throughput(Throughput::Elements(*d as u64));
        group.bench_with_input(BenchmarkId::from_parameter(d), d, |b, _| {
            b.iter(|| {
                black_box(layer.forward(black_box(&input)).unwrap());
            });
        });
    }
    group.finish();
}

/// Benchmark SIMD operations
fn bench_simd_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_ops");

    for size in [256, 512, 1024, 2048].iter() {
        let a = vec![0.5f32; *size];
        let b = vec![0.3f32; *size];

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("dot", size), size, |bench, _| {
            bench.iter(|| {
                black_box(kizzasi_core::simd::dot_product(
                    black_box(&a),
                    black_box(&b),
                ));
            });
        });
    }
    group.finish();
}

/// Benchmark layer normalization
fn bench_layer_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_norm");

    for size in [256, 512, 1024, 2048].iter() {
        let x = Array1::from_elem(*size, 0.5f32);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(kizzasi_core::nn::layer_norm(black_box(&x), black_box(1e-5)));
            });
        });
    }
    group.finish();
}

/// Benchmark softmax
fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    for size in [256, 512, 1024, 2048].iter() {
        let x = Array1::from_elem(*size, 0.5f32);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(kizzasi_core::nn::softmax(black_box(&x)));
            });
        });
    }
    group.finish();
}

/// Benchmark attention mechanisms
fn bench_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention");

    for seq_len in [64, 128, 256].iter() {
        let config = EfficientAttentionConfig {
            num_heads: 8,
            head_dim: 32,
            chunk_size: 64,
            causal: true,
            dropout: 0.0,
        };
        let attention = EfficientMultiHeadAttention::new(config, 256).unwrap();
        let input = Array2::from_elem((*seq_len, 256), 0.5f32);

        group.throughput(Throughput::Elements((seq_len * 256) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), seq_len, |b, _| {
            b.iter(|| {
                black_box(attention.forward(black_box(&input)).unwrap());
            });
        });
    }
    group.finish();
}

/// Benchmark quantization
fn bench_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization");

    for size in [64, 128, 256].iter() {
        let weights = Array2::from_shape_fn((*size, *size), |(i, j)| ((i + j) as f32) * 0.01);

        let quantizer_int8 = DynamicQuantizer::int8_per_tensor();
        let quantizer_int4 = DynamicQuantizer::int4_per_channel();

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::new("int8", size), size, |b, _| {
            b.iter(|| {
                black_box(quantizer_int8.quantize_2d(black_box(&weights)).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("int4", size), size, |b, _| {
            b.iter(|| {
                black_box(quantizer_int4.quantize_2d(black_box(&weights)).unwrap());
            });
        });
    }
    group.finish();
}

/// Benchmark pruning
fn bench_pruning(c: &mut Criterion) {
    let mut group = c.benchmark_group("pruning");

    for size in [64, 128, 256].iter() {
        let weights = Array2::from_shape_fn((*size, *size), |(i, j)| ((i + j) as f32) * 0.01);

        let config_unstructured = PruningConfig::new(PruningStrategy::Magnitude, 0.5);
        let config_structured = PruningConfig::new(PruningStrategy::L2Norm, 0.5)
            .with_granularity(PruningGranularity::Channel);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::new("unstructured", size), size, |b, _| {
            let mut pruner = StructuredPruner::new(config_unstructured.clone()).unwrap();
            b.iter(|| {
                black_box(pruner.prune("layer", black_box(&weights)).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("structured", size), size, |b, _| {
            let mut pruner = StructuredPruner::new(config_structured.clone()).unwrap();
            b.iter(|| {
                black_box(pruner.prune("layer", black_box(&weights)).unwrap());
            });
        });
    }
    group.finish();
}

/// Benchmark batch processing
fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");

    for batch_size in [8, 16, 32].iter() {
        let config = S4DConfig {
            input_dim: 256,
            state_dim: 64,
            hidden_dim: 256,
            delta: 0.001,
            use_hippo: true,
            bidirectional: false,
        };
        let layer = S4DLayer::new(config).unwrap();

        let batch = Array2::from_elem((*batch_size, 256), 0.5f32);

        group.throughput(Throughput::Elements((batch_size * 256) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(layer.forward_sequence(black_box(&batch)).unwrap());
                });
            },
        );
    }
    group.finish();
}

/// Benchmark LoRA operations
fn bench_lora(c: &mut Criterion) {
    let mut group = c.benchmark_group("lora");

    for rank in [4, 8, 16].iter() {
        let config = LoRAConfig::new(*rank, (*rank * 2) as f32);
        let base_weight = Array2::from_elem((256, 256), 0.1f32);
        let layer = LoRALayer::new(config, base_weight).unwrap();

        let input = Array1::from_elem(256, 0.5f32);

        group.throughput(Throughput::Elements(256));
        group.bench_with_input(BenchmarkId::from_parameter(rank), rank, |b, _| {
            b.iter(|| {
                black_box(layer.forward(black_box(&input)).unwrap());
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_ssm_forward,
    bench_mamba2_forward,
    bench_s5_forward,
    bench_simd_ops,
    bench_layer_norm,
    bench_softmax,
    bench_attention,
    bench_quantization,
    bench_pruning,
    bench_batch_processing,
    bench_lora,
);
criterion_main!(benches);
