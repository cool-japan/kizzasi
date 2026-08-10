//! Benchmark: INT8 quantization throughput.
//!
//! Drives [`WeightLoader::quantize_tensor`], [`WeightLoader::dequantize_tensor`],
//! [`WeightLoader::quantize_per_channel`], and
//! [`WeightLoader::dequantize_per_channel`] across a sweep of tensor sizes
//! representative of SSM / transformer weight matrices.
//!
//! Throughput is reported as **elements per second** so the criterion output
//! can be read as "scalar values quantized / dequantized per unit time".

use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kizzasi_core::weights::{WeightLoadConfig, WeightLoader};
use std::hint::black_box;

/// Tensor shapes covering the typical INT8 quantization regime.
///
/// * `64 x 64`   — small projection (e.g. attention head).
/// * `128 x 128` — small SSM hidden layer.
/// * `256 x 256` — medium hidden layer.
/// * `512 x 512` — large hidden layer.
/// * `1024 x 1024` — large weight matrix.
const SHAPES: &[(usize, usize)] = &[(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)];

/// Build a deterministic `Tensor` of the requested shape on CPU.
///
/// Uses `Tensor::randn` for a realistic normal-distribution payload — this
/// matches the kind of data INT8 PTQ is normally evaluated against.
fn make_tensor(rows: usize, cols: usize, device: &Device) -> Tensor {
    Tensor::randn(0.0_f32, 1.0_f32, (rows, cols), device)
        .expect("failed to allocate benchmark tensor")
}

/// Construct a fresh `WeightLoader` with the default `WeightLoadConfig`.
///
/// `WeightLoader::new` takes a `WeightLoadConfig`; there is no `default()`,
/// so the bench drives the public constructor explicitly.
fn make_loader() -> WeightLoader {
    WeightLoader::new(WeightLoadConfig::default())
}

/// Benchmark per-tensor INT8 quantization (`quantize_tensor`).
fn bench_quantize_per_tensor(c: &mut Criterion) {
    let device = Device::Cpu;
    let loader = make_loader();

    let mut group = c.benchmark_group("quantize/per_tensor");
    for &(rows, cols) in SHAPES {
        let n = rows * cols;
        let tensor = make_tensor(rows, cols, &device);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("quantize", format!("{}x{}", rows, cols)),
            &n,
            |b, _| {
                b.iter(|| {
                    loader
                        .quantize_tensor(black_box(&tensor))
                        .expect("quantize_tensor failed")
                });
            },
        );
    }
    group.finish();
}

/// Benchmark per-tensor INT8 dequantization (`dequantize_tensor`).
fn bench_dequantize_per_tensor(c: &mut Criterion) {
    let device = Device::Cpu;
    let loader = make_loader();

    let mut group = c.benchmark_group("quantize/per_tensor_dequant");
    for &(rows, cols) in SHAPES {
        let n = rows * cols;
        let tensor = make_tensor(rows, cols, &device);
        let quantized = loader
            .quantize_tensor(&tensor)
            .expect("pre-quantization for dequant bench failed");
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("dequantize", format!("{}x{}", rows, cols)),
            &n,
            |b, _| {
                b.iter(|| {
                    loader
                        .dequantize_tensor(black_box(&quantized))
                        .expect("dequantize_tensor failed")
                });
            },
        );
    }
    group.finish();
}

/// Benchmark a complete quantize -> dequantize round-trip.
///
/// Useful for understanding the end-to-end latency of post-training
/// quantization passes that go through the WeightLoader API.
fn bench_quantize_roundtrip(c: &mut Criterion) {
    let device = Device::Cpu;
    let loader = make_loader();

    let mut group = c.benchmark_group("quantize/per_tensor_roundtrip");
    for &(rows, cols) in SHAPES {
        let n = rows * cols;
        let tensor = make_tensor(rows, cols, &device);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("roundtrip", format!("{}x{}", rows, cols)),
            &n,
            |b, _| {
                b.iter(|| {
                    let q = loader
                        .quantize_tensor(black_box(&tensor))
                        .expect("quantize_tensor failed");
                    loader
                        .dequantize_tensor(black_box(&q))
                        .expect("dequantize_tensor failed")
                });
            },
        );
    }
    group.finish();
}

/// Benchmark per-channel INT8 quantization across both axes.
///
/// `axis = 0` is the standard "output channel" axis for `[out, in]` weight
/// matrices; `axis = 1` exercises the alternative slicing path (input
/// channels). Both share the implementation but differ in tensor layout, so
/// they tend to have different performance characteristics.
fn bench_quantize_per_channel(c: &mut Criterion) {
    let device = Device::Cpu;
    let loader = make_loader();

    let mut group = c.benchmark_group("quantize/per_channel");
    for &(rows, cols) in SHAPES {
        let n = rows * cols;
        let tensor = make_tensor(rows, cols, &device);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("axis0", format!("{}x{}", rows, cols)),
            &n,
            |b, _| {
                b.iter(|| {
                    loader
                        .quantize_per_channel(black_box(&tensor), 0)
                        .expect("quantize_per_channel(axis=0) failed")
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("axis1", format!("{}x{}", rows, cols)),
            &n,
            |b, _| {
                b.iter(|| {
                    loader
                        .quantize_per_channel(black_box(&tensor), 1)
                        .expect("quantize_per_channel(axis=1) failed")
                });
            },
        );
    }
    group.finish();
}

/// Benchmark per-channel INT8 dequantization.
fn bench_dequantize_per_channel(c: &mut Criterion) {
    let device = Device::Cpu;
    let loader = make_loader();

    let mut group = c.benchmark_group("quantize/per_channel_dequant");
    for &(rows, cols) in SHAPES {
        let n = rows * cols;
        let tensor = make_tensor(rows, cols, &device);
        let quantized = loader
            .quantize_per_channel(&tensor, 0)
            .expect("pre-quantization for per-channel dequant bench failed");
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("dequant_axis0", format!("{}x{}", rows, cols)),
            &n,
            |b, _| {
                b.iter(|| {
                    loader
                        .dequantize_per_channel(black_box(&quantized))
                        .expect("dequantize_per_channel failed")
                });
            },
        );
    }
    group.finish();
}

/// Benchmark quantization across different dtypes.
///
/// `quantize_tensor` casts to `f32` internally before computing min/max, so
/// the input dtype influences the cost of the up-front cast. We exercise
/// both `F32` (no cast) and `F16` (lossy cast) on the same shape.
fn bench_quantize_dtype_sweep(c: &mut Criterion) {
    let device = Device::Cpu;
    let loader = make_loader();
    let shape = (256usize, 256usize);
    let n = shape.0 * shape.1;

    let mut group = c.benchmark_group("quantize/dtype_sweep");
    group.throughput(Throughput::Elements(n as u64));

    let tensor_f32 = make_tensor(shape.0, shape.1, &device);
    group.bench_function(BenchmarkId::new("dtype", "f32"), |b| {
        b.iter(|| {
            loader
                .quantize_tensor(black_box(&tensor_f32))
                .expect("quantize f32 failed")
        });
    });

    let tensor_f16 = tensor_f32.to_dtype(DType::F16).expect("cast to f16 failed");
    group.bench_function(BenchmarkId::new("dtype", "f16"), |b| {
        b.iter(|| {
            loader
                .quantize_tensor(black_box(&tensor_f16))
                .expect("quantize f16 failed")
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_quantize_per_tensor,
    bench_dequantize_per_tensor,
    bench_quantize_roundtrip,
    bench_quantize_per_channel,
    bench_dequantize_per_channel,
    bench_quantize_dtype_sweep,
);
criterion_main!(benches);
