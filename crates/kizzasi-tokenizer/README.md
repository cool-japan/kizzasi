# kizzasi-tokenizer

Signal quantization and tokenization for Kizzasi AGSP.

## Overview

Comprehensive tokenization toolkit for continuous signals with VQ-VAE, μ-law, and advanced quantization strategies. Designed for audio, sensors, and general signal compression.

## Features

- **VQ-VAE**: Vector quantization with EMA updates and residual VQ
- **μ-law Codec**: 8-bit and 16-bit compression with expansion
- **Advanced Quantizers**: Adaptive, dead-zone, non-uniform, Lloyd-Max
- **Specialized**: Wavelet, DCT, Fourier, k-means tokenizers
- **Neural Codec**: SoundStream/Encodec-style architecture
- **Domain-Specific**: Speech, music, environmental audio tokenizers
- **GPU Acceleration**: CUDA/Metal support for batch operations
- **SIMD Optimized**: 8-way vectorization for quantization

## Quick Start

```rust
use kizzasi_tokenizer::{LinearQuantizer, SignalTokenizer};

// 8-bit linear quantization
let mut quantizer = LinearQuantizer::new(8, -1.0, 1.0)?;

let signal = Array1::from_vec(vec![0.5, -0.3, 0.8]);
let codes = quantizer.encode(&signal)?;
let reconstructed = quantizer.decode(&codes)?;

// VQ-VAE with learned codebook
use kizzasi_tokenizer::VQVAETokenizer;
let vqvae = VQVAETokenizer::new(512, 32, 64)?; // codebook_size, dim, embed_dim
```

## Compression Performance

- μ-law: 4x-8x compression, <1ms latency
- VQ-VAE: 10x-100x compression, learned representations
- Neural Codec: 20x-200x compression, high quality

## Documentation

- [API Documentation](https://docs.rs/kizzasi-tokenizer)
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under the Apache License, Version 2.0.
