//! Perceptual (Bark-band, psychoacoustic) quantization demo.
//!
//! Encodes a 440 Hz tone (and a multi-tone chord) with
//! [`PerceptualQuantizer`], reports the resulting bitrate and the
//! reconstruction SNR, and prints the per-Bark-band absolute threshold of
//! hearing so the reader can see which bands receive (essentially) no bits.
//!
//! Run with:
//! ```bash
//! cargo run --example perceptual_quant -p kizzasi-tokenizer
//! ```

use kizzasi_tokenizer::perceptual::{absolute_threshold_db, PerceptualQuantizer};
use scirs2_core::ndarray::Array1;

fn synth_tone(freq_hz: f32, sample_rate: f32, num_samples: usize) -> Array1<f32> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let samples: Vec<f32> = (0..num_samples)
        .map(|i| (two_pi * freq_hz * i as f32 / sample_rate).sin())
        .collect();
    Array1::from(samples)
}

fn snr_db(original: &Array1<f32>, reconstructed: &Array1<f32>) -> f32 {
    // Align lengths in case overlap-add trimmed/padded the output.
    let n = original.len().min(reconstructed.len());
    if n == 0 {
        return f32::NEG_INFINITY;
    }

    let mut signal_power = 0.0_f64;
    let mut noise_power = 0.0_f64;
    for i in 0..n {
        let s = original[i] as f64;
        let r = reconstructed[i] as f64;
        signal_power += s * s;
        noise_power += (s - r) * (s - r);
    }
    signal_power /= n as f64;
    noise_power /= n as f64;

    if noise_power < 1e-20 {
        return f32::INFINITY;
    }
    (10.0 * (signal_power / noise_power).log10()) as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Perceptual (Bark-band) Quantizer Demo ===\n");

    let sample_rate = 44_100.0_f32;
    let frame_size = 1024_usize;
    let total_bits_per_frame = 256_usize;

    let quant = PerceptualQuantizer::new(sample_rate, frame_size, total_bits_per_frame)?;

    println!(
        "Sample rate    : {} Hz\nFrame size     : {} samples\nBits per frame : {}\nBark bands     : {}\n",
        sample_rate, frame_size, total_bits_per_frame, quant.bark_bands.num_bands
    );

    // Show the absolute threshold of hearing per Bark-band centre.
    println!("Per-band absolute threshold (Terhardt, dB SPL):");
    for (b, &centre_hz) in quant.bark_bands.centers_hz.iter().enumerate() {
        let t = absolute_threshold_db(centre_hz);
        println!(
            "  Band {:>2} | centre {:>7.1} Hz | threshold {:>6.1} dB",
            b, centre_hz, t
        );
    }
    println!();

    // 1) Pure 440 Hz tone (A4) of length 2 frames worth of samples.
    let tone = synth_tone(440.0, sample_rate, 2 * frame_size);
    let tokens = quant.encode(&tone)?;
    let recon = quant.decode(&tokens)?;
    let snr = snr_db(&tone, &recon);
    let bitrate_kbps =
        total_bits_per_frame as f32 * sample_rate / (frame_size as f32 * 1000.0_f32 / 2.0);
    println!(
        "Single 440 Hz tone : {:>5} tokens -> SNR = {:>6.2} dB at ~{:.1} kbps",
        tokens.len(),
        snr,
        bitrate_kbps
    );

    // 2) A major-triad-ish chord: 440 Hz + 554 Hz (≈ C#5) + 659 Hz (≈ E5).
    let mut chord = synth_tone(440.0, sample_rate, 2 * frame_size);
    let third = synth_tone(554.37, sample_rate, 2 * frame_size);
    let fifth = synth_tone(659.25, sample_rate, 2 * frame_size);
    for i in 0..chord.len() {
        chord[i] = (chord[i] + third[i] + fifth[i]) / 3.0;
    }
    let chord_tokens = quant.encode(&chord)?;
    let chord_recon = quant.decode(&chord_tokens)?;
    let chord_snr = snr_db(&chord, &chord_recon);
    println!(
        "A-major-ish chord  : {:>5} tokens -> SNR = {:>6.2} dB",
        chord_tokens.len(),
        chord_snr,
    );

    println!(
        "\nNote: phase information is discarded (zero-phase reconstruction), so SNR\nis lower than a lossless codec would deliver. Bit allocation is concentrated\non bands whose energy exceeds the Terhardt absolute threshold."
    );

    Ok(())
}
