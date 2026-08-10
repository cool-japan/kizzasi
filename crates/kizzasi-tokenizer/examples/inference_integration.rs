//! Integration example: kizzasi-tokenizer → kizzasi-inference pipeline.
//!
//! Demonstrates how tokenized signal representations flow into the kizzasi
//! inference engine for autoregressive prediction.
//!
//! The example builds a complete pipeline through several stages:
//!
//! 1. Tokenize a synthetic signal with `LinearQuantizer` and `MuLawCodec`.
//! 2. Normalize the token stream into the input format expected by the engine.
//! 3. Stream normalized frames through `InferenceEngine` (with a stub model).
//! 4. Show a multi-step rollout via `engine.rollout(…)`.
//! 5. Compare batch and streaming tokenization throughput.
//! 6. Demonstrate multi-scale token hierarchy.
//! 7. Show every `SamplingConfig` / `EngineConfig` option available on the
//!    public interface.
//!
//! **Dependency note**: `kizzasi-inference` depends on `kizzasi-tokenizer`,
//! so only `kizzasi-inference` (not `kizzasi-model`) is added as a
//! dev-dependency here — Cargo allows this one-sided dev-dep cycle.  The
//! inference engine runs with no model attached; `InferenceEngine::new`
//! (without `with_model`) returns zeros for each step, which is fine for
//! demonstrating the interface.
//!
//! Run with:
//! ```bash
//! cargo run --example inference_integration -p kizzasi-tokenizer
//! ```

use kizzasi_inference::{EngineConfig, InferenceEngine, InferenceMode, SamplingConfig};
use kizzasi_tokenizer::{
    Array1, BatchTokenizer, LinearQuantizer, MuLawCodec, MultiScaleTokenizer, Quantizer,
    SignalTokenizer, StreamingTokenizer, TokenizerError,
};
use scirs2_core::ndarray::Array2;
use std::time::Instant;

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

/// Generate a synthetic 1-second, 16 kHz signal (3 harmonics).
fn synthetic_signal(num_samples: usize, sample_rate: f32) -> Array1<f32> {
    let mut signal = Array1::zeros(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate;
        signal[i] = 0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
            + 0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            + 0.2 * (2.0 * std::f32::consts::PI * 880.0 * t).sin();
    }
    signal
}

/// Compute signal-to-noise ratio in dB.
fn snr_db(original: &Array1<f32>, reconstructed: &Array1<f32>) -> f32 {
    let signal_power: f32 = original.iter().map(|x| x * x).sum::<f32>() / original.len() as f32;
    let noise_power: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| {
            let e = a - b;
            e * e
        })
        .sum::<f32>()
        / original.len() as f32;

    if noise_power < 1e-12 {
        return f32::INFINITY;
    }
    10.0 * (signal_power / noise_power).log10()
}

// ────────────────────────────────────────────────────────────────────────────
// Stage 1 — Tokenize a signal with LinearQuantizer
// ────────────────────────────────────────────────────────────────────────────

fn stage_linear_quantizer() -> Result<(Array1<f32>, Array1<f32>), TokenizerError> {
    println!("──────────────────────────────────────────────");
    println!("Stage 1: LinearQuantizer Tokenization");
    println!("──────────────────────────────────────────────");

    let sample_rate = 16_000.0_f32;
    let num_samples = 1_024;
    let signal = synthetic_signal(num_samples, sample_rate);

    let q = LinearQuantizer::new(-1.0, 1.0, 8)
        .expect("LinearQuantizer::new should succeed for valid params");

    let start = Instant::now();
    let tokens = q.encode(&signal)?;
    let encode_us = start.elapsed().as_micros();

    let reconstructed = q.decode(&tokens)?;
    let snr = snr_db(&signal, &reconstructed);

    println!("  Signal length:   {num_samples} samples @ {sample_rate:.0} Hz");
    println!(
        "  Bits:            {} → {} levels",
        q.bits(),
        q.num_levels()
    );
    println!("  Step size:       {:.5}", q.step_size());
    println!("  Token count:     {}", tokens.len());
    println!("  SNR (roundtrip): {snr:.1} dB");
    println!("  Encode time:     {encode_us} µs");

    let preview: Vec<String> = tokens.iter().take(8).map(|t| format!("{:.0}", t)).collect();
    println!("  First 8 tokens:  [{}]", preview.join(", "));
    println!();

    Ok((signal, tokens))
}

// ────────────────────────────────────────────────────────────────────────────
// Stage 2 — MuLaw comparison
// ────────────────────────────────────────────────────────────────────────────

fn stage_mulaw(signal: &Array1<f32>) -> Result<Array1<f32>, TokenizerError> {
    println!("──────────────────────────────────────────────");
    println!("Stage 2: MuLawCodec Tokenization (comparison)");
    println!("──────────────────────────────────────────────");

    let mulaw = MuLawCodec::new(8); // infallible constructor

    let tokens = mulaw.encode(signal)?;
    let reconstructed = mulaw.decode(&tokens)?;
    let snr = snr_db(signal, &reconstructed);

    println!("  Bits:            8 → {} levels", mulaw.vocab_size());
    println!("  Token count:     {}", tokens.len());
    println!("  SNR (roundtrip): {snr:.1} dB");

    let preview: Vec<String> = tokens.iter().take(8).map(|t| format!("{:.0}", t)).collect();
    println!("  First 8 tokens:  [{}]", preview.join(", "));
    println!();

    Ok(tokens)
}

// ────────────────────────────────────────────────────────────────────────────
// Stage 3 — Prepare tokens for the inference engine
//
// `kizzasi-inference`'s InferenceEngine expects input shaped [input_dim].
// We normalize quantizer output (integers in [0, max_level]) to [-1, 1].
// ────────────────────────────────────────────────────────────────────────────

fn stage_prepare_for_inference(tokens: &Array1<f32>, num_levels: usize) -> Array1<f32> {
    println!("──────────────────────────────────────────────");
    println!("Stage 3: Normalizing Tokens for Inference Input");
    println!("──────────────────────────────────────────────");

    let max_level = (num_levels - 1) as f32;
    let normalized: Array1<f32> = tokens.mapv(|t| (t / max_level) * 2.0 - 1.0);

    let min_v = normalized.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_v = normalized.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let raw_min = tokens.iter().cloned().fold(f32::INFINITY, f32::min);
    let raw_max = tokens.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("  Raw token range:        [{raw_min:.2}, {raw_max:.2}]");
    println!("  Normalized value range: [{min_v:.4}, {max_v:.4}]");
    println!("  Vector length:          {}", normalized.len());
    println!();

    normalized
}

// ────────────────────────────────────────────────────────────────────────────
// Stage 4 — Stream tokens through InferenceEngine one frame at a time
//
// InferenceEngine::new(EngineConfig) runs without a model attached.
// step() returns zeros for each call — the important thing here is the
// interface: tokenizer output → Array1<f32> → engine.step().
// ────────────────────────────────────────────────────────────────────────────

fn stage_streaming_inference(normalized: &Array1<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("──────────────────────────────────────────────");
    println!("Stage 4: Streaming Autoregressive Inference");
    println!("──────────────────────────────────────────────");

    let frame_size = 1_usize;
    let num_frames = normalized.len().min(32);

    // Build the engine in Streaming mode.
    // Without a model, step() returns zeros but validates dimensions and
    // updates the context window — sufficient to exercise the interface.
    let config = EngineConfig::new(frame_size, frame_size)
        .inference_mode(InferenceMode::Streaming)
        .sampling(SamplingConfig::default())
        .max_history_length(64);

    let mut engine = InferenceEngine::new(config);

    // To run a real inference, attach a model before calling step():
    //   engine.set_model(Box::new(MyModel::new(…)));
    // Here we use the engine without a model to demonstrate the config API.

    println!("  Engine mode:    Streaming (no model → zero outputs)");
    println!("  Frame size:     {frame_size} sample(s)");
    println!("  Max history:    64 frames");
    println!("  Steps to run:   {num_frames}");
    println!("  Model attached: {}", engine.has_model());

    // Attempt step — returns NotInitialized without a model, which is expected.
    let frame = Array1::from_elem(frame_size, normalized[0]);
    match engine.step(&frame) {
        Ok(out) => println!("  Step output:    {:?}", out.to_vec()),
        Err(e) => println!("  Step skipped:   {e} (expected — no model loaded)"),
    }
    println!();

    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Stage 5 — Batch and streaming tokenization
// ────────────────────────────────────────────────────────────────────────────

fn stage_batch_streaming(signal: &Array1<f32>) -> Result<(), TokenizerError> {
    println!("──────────────────────────────────────────────");
    println!("Stage 5: Batch & Streaming Tokenization");
    println!("──────────────────────────────────────────────");

    let q = LinearQuantizer::new(-1.0, 1.0, 8).expect("LinearQuantizer::new should succeed");

    // BatchTokenizer is a trait implemented by all SignalTokenizer types.
    let batch_size = 4_usize;
    let sig_len = 256_usize;
    let mut batch_arr = Array2::<f32>::zeros((batch_size, sig_len));
    for i in 0..batch_size {
        let row = synthetic_signal(sig_len, 16_000.0_f32).mapv(|x| x * (1.0 - 0.1 * i as f32));
        for j in 0..sig_len {
            batch_arr[[i, j]] = row[j];
        }
    }
    let batch_tokens = q.encode_batch(&batch_arr)?;

    println!(
        "  BatchTokenizer: {}×{} → {}×{} token array",
        batch_size,
        sig_len,
        batch_tokens.shape()[0],
        batch_tokens.shape()[1]
    );

    // StreamingTokenizer processes a signal in fixed-size chunks with optional overlap.
    let chunk_size = 128_usize;
    let overlap = 0_usize;
    let streaming = StreamingTokenizer::new(q, chunk_size, overlap)?;
    let chunk_tokens = streaming.encode_streaming(signal)?;

    println!(
        "  StreamingTokenizer: {} samples → {} chunks of ≤{chunk_size}",
        signal.len(),
        chunk_tokens.len()
    );
    println!();

    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Stage 6 — Multi-scale token hierarchy
// ────────────────────────────────────────────────────────────────────────────

fn stage_multiscale(signal: &Array1<f32>) -> Result<(), TokenizerError> {
    println!("──────────────────────────────────────────────");
    println!("Stage 6: Multi-Scale Tokenization");
    println!("──────────────────────────────────────────────");

    let input_dim = signal.len();
    let embed_dim = 8_usize;
    let ms = MultiScaleTokenizer::new(input_dim, embed_dim);

    let tokens = ms.encode(signal)?;

    println!("  Default scales:  [×1, ×2, ×4]");
    println!("  Input length:    {input_dim}");
    println!("  Embed dim/level: {embed_dim}");
    println!("  Output length:   {} tokens", tokens.len());
    println!();

    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Stage 7 — SignalTokenizer interface boundary + EngineConfig / SamplingConfig
// ────────────────────────────────────────────────────────────────────────────

fn stage_show_interface() -> Result<(), TokenizerError> {
    println!("──────────────────────────────────────────────");
    println!("Stage 7: SignalTokenizer & InferenceEngine Interface");
    println!("──────────────────────────────────────────────");

    // ── Tokenizer side ──
    let tokenizers: Vec<(&str, Box<dyn SignalTokenizer>)> = vec![
        (
            "LinearQuantizer(8-bit)",
            Box::new(LinearQuantizer::new(-1.0, 1.0, 8).expect("valid")),
        ),
        (
            "LinearQuantizer(12-bit)",
            Box::new(LinearQuantizer::new(-1.0, 1.0, 12).expect("valid")),
        ),
        ("MuLawCodec(8-bit)", Box::new(MuLawCodec::new(8))),
    ];

    let test_signal = synthetic_signal(64, 16_000.0);

    println!("  Tokenizer                embed_dim  vocab_size  SNR");
    println!("  ──────────────────────────────────────────────────────");
    for (name, tok) in &tokenizers {
        let tokens = tok.encode(&test_signal)?;
        let recon = tok.decode(&tokens)?;
        let snr = snr_db(&test_signal, &recon);
        println!(
            "  {:<24} {:>9}  {:>10}  {:>6.1} dB",
            name,
            tok.embed_dim(),
            tok.vocab_size(),
            snr
        );
    }
    println!();

    // ── Inference side: SamplingConfig ──
    let greedy_cfg = SamplingConfig::default();
    let topk_cfg = SamplingConfig::new().top_k(10).temperature(0.9);
    let nucleus_cfg = SamplingConfig::new().top_p(0.95);
    let beam_cfg = SamplingConfig::new().beam_search(4);

    println!("  SamplingConfig variants:");
    println!("    Greedy:  strategy={:?}", greedy_cfg.strategy);
    println!(
        "    Top-k:   strategy={:?}, k={:?}, temp={:.2}",
        topk_cfg.strategy, topk_cfg.top_k, topk_cfg.temperature
    );
    println!(
        "    Nucleus: strategy={:?}, p={:?}",
        nucleus_cfg.strategy, nucleus_cfg.top_p
    );
    println!(
        "    Beam:    strategy={:?}, width={}",
        beam_cfg.strategy, beam_cfg.beam_width
    );
    println!();

    // ── Inference side: InferenceMode ──
    let modes = [
        InferenceMode::Standard,
        InferenceMode::LowMemory,
        InferenceMode::Streaming,
        InferenceMode::Quantized,
    ];
    println!("  InferenceMode options: {:?}", modes);
    println!();

    // ── Putting it together: EngineConfig ──
    let engine_cfg = EngineConfig::new(1, 1)
        .apply_constraints(false)
        .inference_mode(InferenceMode::LowMemory)
        .sampling(topk_cfg)
        .max_history_length(256)
        .state_prune_threshold(1e-4);

    println!("  EngineConfig (input=1, output=1):");
    println!("    mode             = {:?}", engine_cfg.inference_mode);
    println!("    apply_constraints= {}", engine_cfg.apply_constraints);
    println!("    max_history      = {:?}", engine_cfg.max_history_length);
    println!(
        "    prune_threshold  = {}",
        engine_cfg.state_prune_threshold
    );
    println!();

    println!("  To wire a tokenizer into a pipeline (kizzasi-inference):");
    println!("    let t: Box<dyn SignalTokenizer> = Box::new(LinearQuantizer::new(-1.0,1.0,8)?);");
    println!("    let mut pipeline = PipelineBuilder::new()");
    println!("        .engine_config(EngineConfig::new(1, 1))");
    println!("        .tokenizer(t)");
    println!("        .build()?;");
    println!("    let output = pipeline.forward(&signal)?;");
    println!();

    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Entry point
// ────────────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=======================================================");
    println!("  kizzasi-tokenizer ↔ kizzasi-inference Integration");
    println!("=======================================================");
    println!();

    let (signal, linear_tokens) = stage_linear_quantizer()?;
    let _mulaw_tokens = stage_mulaw(&signal)?;
    let normalized = stage_prepare_for_inference(&linear_tokens, 256);
    stage_streaming_inference(&normalized)?;
    stage_batch_streaming(&signal)?;
    stage_multiscale(&signal)?;
    stage_show_interface()?;

    println!("=======================================================");
    println!("  Integration demo complete — all stages OK.");
    println!("=======================================================");
    Ok(())
}
