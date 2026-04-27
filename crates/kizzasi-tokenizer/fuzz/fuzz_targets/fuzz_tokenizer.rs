#![no_main]

use kizzasi_tokenizer::{
    LinearQuantizer, MuLawCodec, Quantizer, SignalTokenizer,
};

#[cfg(feature = "vqvae")]
use kizzasi_tokenizer::{VQConfig, VectorQuantizer};

use kizzasi_tokenizer::Array1;
use libfuzzer_sys::fuzz_target;

/// Interpret a byte slice as a sequence of f32 values.
/// If the slice length is not a multiple of 4, the trailing bytes are ignored.
fn bytes_to_f32_vec(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| {
            let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(arr)
        })
        .collect()
}

fuzz_target!(|data: &[u8]| {
    // Require at least a few bytes to derive meaningful parameters.
    if data.len() < 4 {
        return;
    }

    // Use the first byte to derive a bit-depth in the range [1, 16].
    let bits_raw = data[0];
    // Map 0..=255 → 1..=16
    let bits: u8 = (bits_raw % 16) + 1;

    // Remaining bytes become the signal samples.
    let samples = bytes_to_f32_vec(&data[1..]);

    // -------------------------------------------------------------------------
    // Exercise MuLawCodec
    // -------------------------------------------------------------------------
    let mulaw = MuLawCodec::new(bits);

    // Single-sample encode / decode round-trip
    for &raw in data.iter().take(16) {
        // Map byte to [-1.0, 1.0]
        let sample = (raw as f32 / 127.5) - 1.0;
        let level = mulaw.quantize(sample);
        let _recovered = mulaw.dequantize(level);
    }

    // Array encode / decode (if we have samples)
    if !samples.is_empty() {
        let signal = Array1::from_vec(samples.clone());
        if let Ok(encoded) = mulaw.encode(&signal) {
            let _ = mulaw.decode(&encoded);
        }
    }

    // -------------------------------------------------------------------------
    // Exercise LinearQuantizer
    // -------------------------------------------------------------------------
    // Derive min/max from the second and third bytes so they are well-defined.
    let byte1 = data[1 % data.len()] as f32;
    let byte2 = data[2 % data.len()] as f32;
    // Ensure min < max by construction.
    let (min_val, max_val) = if byte1 < byte2 {
        (byte1 - 128.0, byte2 - 128.0 + 1.0)
    } else {
        (byte2 - 128.0, byte1 - 128.0 + 1.0)
    };

    if let Ok(lq) = LinearQuantizer::new(min_val, max_val, bits) {
        // Single-value round-trips
        for &raw in data.iter().take(16) {
            let value = (raw as f32 / 255.0) * (max_val - min_val) + min_val;
            let level = lq.quantize(value);
            let _recovered = lq.dequantize(level);
        }

        // Array encode / decode
        if !samples.is_empty() {
            let signal = Array1::from_vec(samples.clone());
            if let Ok(encoded) = lq.encode(&signal) {
                let _ = lq.decode(&encoded);
            }
        }
    }

    // Normalized variant (always valid for bits 1..=16)
    if let Ok(lq_norm) = LinearQuantizer::normalized(bits) {
        if !samples.is_empty() {
            let signal = Array1::from_vec(samples.clone());
            if let Ok(encoded) = lq_norm.encode(&signal) {
                let _ = lq_norm.decode(&encoded);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Exercise VectorQuantizer (vqvae feature)
    // -------------------------------------------------------------------------
    #[cfg(feature = "vqvae")]
    {
        // Use a small codebook / embed_dim to keep fuzzing tractable.
        let codebook_size: usize = ((data[0] as usize) % 16) + 2; // 2..=17
        let embed_dim: usize = ((data[1 % data.len()] as usize) % 8) + 1; // 1..=8

        let config = VQConfig {
            codebook_size,
            embed_dim,
            commitment_beta: 0.25,
            ema_decay: 0.99,
            epsilon: 1e-5,
            use_ema: true,
        };

        let vq = VectorQuantizer::new(config);

        // Build an input that is a multiple of embed_dim in length.
        if samples.len() >= embed_dim {
            let trimmed_len = (samples.len() / embed_dim) * embed_dim;
            let input = Array1::from_vec(samples[..trimmed_len].to_vec());
            // encode uses SignalTokenizer; the output shape equals the input shape.
            if let Ok(encoded) = vq.encode(&input) {
                let _ = vq.decode(&encoded);
            }
        }
    }
});
