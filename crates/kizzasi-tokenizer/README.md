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
- **Speaker-Aware Tokenization**: Multi-speaker support with EMA-updated k-means++ codebook
- **Psychoacoustic Bit Allocation**: Perceptual quantizer allocates bits per Bark critical band proportional to masking exceedance, achieving perceptual transparency at low bitrates
- **Pure-Rust FFT**: Hann-windowed STFT/ISTFT via oxifft (no C/Fortran) for perceptual quantization
- **PEAQ Quality Evaluation**: ITU-R BS.1387-1 Basic Model — 11 MOVs, Distortion Index, and ODG in `[-4, 0]` from reference + test signals
- **GPU Acceleration**: CUDA/Metal support for batch operations
- **SIMD Optimized**: 8-way vectorization for quantization

## Tokenizer Types

### Core / General Purpose

| Type | Description |
|---|---|
| `LinearQuantizer` | Uniform N-bit scalar quantization |
| `MuLawCodec` | μ-law companding (8-bit or 16-bit) |
| `AdaptiveQuantizer` | Range-adaptive scalar quantizer |
| `DeadZoneQuantizer` | Dead-zone quantizer for sparse signals |
| `NonUniformQuantizer` | Non-uniform (Lloyd-Max) quantization |

### Structural / Hierarchical

| Type | Description |
|---|---|
| `MultiScaleTokenizer` | Multi-resolution hierarchical tokenization |
| `PyramidTokenizer` | Pyramid tokenization with residual encoding |
| `HierarchicalTokenizer` | Tree-structured hierarchical codebook |

### Spectral / Transform

| Type | Description |
|---|---|
| `WaveletTokenizer` | Wavelet-domain tokenization |
| `DCTTokenizer` | DCT-domain tokenization |
| `FourierTokenizer` | Fourier-domain tokenization |
| `KMeansTokenizer` | k-means vector quantization |

### Neural / Learned

| Type | Description |
|---|---|
| `VQVAETokenizer` | VQ-VAE with straight-through estimator |
| `ResidualVQ` | Residual vector quantization |
| `ProductQuantizer` | Product quantization |
| `NeuralCodec` | SoundStream/Encodec-style neural codec |
| `TransformerTokenizer` | Transformer-based tokenization |

### Domain-Specific

| Type | Description |
|---|---|
| `SpeechTokenizer` | Mel-spectrogram-based speech tokenizer |
| `MusicTokenizer` | Music-optimized tokenizer |
| `EnvironmentalTokenizer` | Environmental audio tokenizer |

### v0.2.2: Multi-Speaker and Perceptual

| Type | Description |
|---|---|
| `MultiSpeakerTokenizer` | Multi-speaker acoustic tokenizer with k-means++ speaker codebook and EMA updates; supports explicit speaker encoding via `encode_with_speaker(signal, speaker_id)`, blind speaker inference via `encode_blind(signal)`, deterministic mel-spectrogram decode via `decode(token)`, and voice re-targeting via `re_target(token, target_speaker)`. Returns `MultiSpeakerToken { acoustic, speaker_id, num_frames, n_mels, mel_min, mel_max }`. Implements `SignalTokenizer`. |
| `PerceptualQuantizer` | Bark-scale psychoacoustic quantizer based on Zwicker (1980) 24 critical bands, Traunmüller frequency-to-Bark conversion, and Terhardt absolute-threshold-of-hearing model. Allocates bits per band proportional to masking exceedance. Uses Hann-windowed STFT/ISTFT via oxifft (pure Rust). Implements `SignalTokenizer`. |
| `PeaqEvaluator` | ITU-R BS.1387-1 Basic Model perceptual audio quality evaluator. Given a reference signal and a degraded test signal, returns 11 Model Output Variables (MOVs), a Distortion Index (DI), and an Objective Difference Grade (ODG) in `[-4, 0]`. Implements the full ear model: 109-band Bark grouping, W(f) outer/middle-ear weighting, level-dependent frequency spreading, and IIR time spreading. ODG grades: `Imperceptible` (0 to −0.5) → `VeryAnnoying` (< −3.5). Note: ITU-conformant in shape; `WEIGHTS_VERIFIED = false` until paid BS.1387-1 Annex 2 test vectors are sourced. |

#### Supporting types re-exported from `multi_speaker` and `perceptual`

- `SpeakerCodebook` — EMA-updated centroid matrix; k-means++ initialisation
- `MultiSpeakerConfig` — configuration struct for `MultiSpeakerTokenizer`
- `MultiSpeakerToken` — self-contained acoustic + identity token (see above)
- `BarkBands` — Bark-scale critical-band descriptor (standard 24-band layout)
- `frequency_to_bark(hz)` — Traunmüller formula (free function)
- `absolute_threshold_db(hz)` — Terhardt ATH model (free function)
- `PeaqConfig` — frame size, sample rate, hop size, num_bark_bands
- `PeaqResult` — `{ movs: PeaqMovs, distortion_index: f32, odg: f32, grade: OdgGrade }`
- `OdgGrade` — enum with 5 grades from `Imperceptible` to `VeryAnnoying`

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

### Multi-Speaker Tokenization (v0.2.2)

```rust
use kizzasi_tokenizer::{MultiSpeakerConfig, MultiSpeakerTokenizer};

let config = MultiSpeakerConfig::default(); // 16 speakers, 8-bit mel quantization
let mut tok = MultiSpeakerTokenizer::new(config)?;

// Fit the speaker codebook on a set of training utterances
tok.fit_speakers(&utterances)?;

// Encode with an explicit speaker ID
let token = tok.encode_with_speaker(&signal, 3)?;

// Or let the tokenizer infer the speaker automatically
let token = tok.encode_blind(&signal)?;

// Decode to a flat mel spectrogram (not audio — use an external vocoder)
let mel_flat = tok.decode(&token)?;

// Re-target the acoustic content to a different speaker identity
let re_targeted = tok.re_target(&token, 7)?;
```

### Perceptual Quantization (v0.2.2)

```rust
use kizzasi_tokenizer::PerceptualQuantizer;

// 16 kHz audio, 1024-sample frames, 256 bits per frame across 24 Bark bands
let pq = PerceptualQuantizer::new(16000.0, 1024, 256)?;

let tokens = pq.encode(&signal)?;          // Vec<u32> packed token stream
let reconstructed = pq.decode(&tokens)?;  // overlap-add synthesis
```

### PEAQ Quality Evaluation (v0.2.2)

```rust
use kizzasi_tokenizer::{PeaqConfig, PeaqEvaluator};
use scirs2_core::ndarray::Array1;

let config = PeaqConfig::default(); // 48 kHz, 2048-sample frames, 109 Bark bands
let mut evaluator = PeaqEvaluator::new(config)?;

// 1 s reference at 48 kHz
let n = 48_000;
let reference: Array1<f32> = Array1::from_iter(
    (0..n).map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48_000.0).sin())
);
let test = reference.clone(); // identical → should grade as Imperceptible

let result = evaluator.evaluate(&reference, &test)?;
println!("ODG = {:.2}  ({:?})", result.odg, result.grade);
// ODG ≈ 0.0 (Imperceptible) for identical signals with placeholder weights
```

## Compression Performance

- μ-law: 4x-8x compression, <1ms latency
- VQ-VAE: 10x-100x compression, learned representations
- Neural Codec: 20x-200x compression, high quality
- `MultiSpeakerTokenizer`: lossless up to quantization error; 8-bit default
- `PerceptualQuantizer`: bit rate controlled per-frame; perceptual transparency tunable via `total_bits_per_frame`

## Test Coverage

445 tests across all tokenizer types and supporting utilities.

## Documentation

- [API Documentation](https://docs.rs/kizzasi-tokenizer)
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under the Apache License, Version 2.0.
