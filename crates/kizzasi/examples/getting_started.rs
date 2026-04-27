//! # Getting Started with Kizzasi
//!
//! This example walks through the core Kizzasi workflow:
//!
//! 1. **Tokenization** — Converting continuous signals to discrete representations
//!    using `LinearQuantizer` (uniform quantization) and `MuLawCodec` (logarithmic
//!    companding, common in telephony and audio).
//!
//! 2. **Inference** — Building a small autoregressive predictor with a Mamba2 backend,
//!    running single-step and multi-step predictions via the `SignalPredictor` trait.
//!
//! 3. **Constraint Enforcement** — Installing guardrails that project predictions
//!    into a safe operating region (e.g., joint-angle limits for a robot arm).
//!
//! Run with:
//! ```bash
//! cargo run --example getting_started -p kizzasi
//! ```

use kizzasi::prelude::*;
use kizzasi_tokenizer::{LinearQuantizer, MuLawCodec, Quantizer, SignalTokenizer};
use scirs2_core::ndarray::Array1;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Kizzasi Getting Started Example ===\n");

    // Step 1: Explore signal tokenization strategies
    example_tokenization()?;

    // Step 2: Build a predictor and run inference
    example_inference()?;

    // Step 3: Enforce safety constraints on predictions
    example_constraints()?;

    println!("\n=== All examples completed successfully! ===");
    Ok(())
}

/// Demonstrates linear and mu-law quantization of a continuous signal.
fn example_tokenization() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("--- Tokenization ---");

    // ---- Linear Quantizer -----------------------------------------------
    // Maps the range [-1.0, 1.0] uniformly onto 256 levels (8 bits).
    let quantizer = LinearQuantizer::new(-1.0, 1.0, 8)?;

    println!(
        "LinearQuantizer: {} levels, step = {:.6}",
        quantizer.num_levels(),
        quantizer.step_size(),
    );

    // Quantize individual samples using the Quantizer trait
    let samples = [0.0f32, 0.5, -0.5, 1.0, -1.0];
    for &s in &samples {
        let level = quantizer.quantize(s);
        let recovered = quantizer.dequantize(level);
        println!(
            "  sample {:+.2} -> level {:3} -> recovered {:+.4}",
            s, level, recovered
        );
    }

    // Encode/decode an entire array using SignalTokenizer trait
    let signal = Array1::from_vec(samples.to_vec());
    let encoded = quantizer.encode(&signal)?;
    let decoded = quantizer.decode(&encoded)?;
    println!(
        "Batch encode -> decode error (max): {:.6}",
        signal
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    );

    // ---- Mu-Law Codec ---------------------------------------------------
    // Logarithmic companding; provides perceptually uniform quantization
    // for audio (telephone standard, WaveNet, etc.).
    let mulaw = MuLawCodec::new(8);

    println!(
        "\nMuLawCodec: mu = {:.1}, {} levels",
        mulaw.mu(),
        mulaw.vocab_size(),
    );

    for &s in &samples {
        let level = mulaw.quantize(s);
        let recovered = mulaw.dequantize(level);
        println!(
            "  sample {:+.2} -> level {:3} -> recovered {:+.4}",
            s, level, recovered
        );
    }

    // SignalTokenizer encode/decode for a full array
    let encoded_mu = mulaw.encode(&signal)?;
    let decoded_mu = mulaw.decode(&encoded_mu)?;
    println!(
        "MuLaw encode -> decode error (max): {:.6}",
        signal
            .iter()
            .zip(decoded_mu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    );

    Ok(())
}

/// Demonstrates building a predictor and running single- and multi-step inference.
fn example_inference() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Model Inference ---");

    // Build a compact Mamba2 predictor with 3-dimensional input/output.
    // hidden_dim and state_dim are kept small for a fast demo.
    let mut predictor = KizzasiBuilder::new()
        .model_type(ModelType::Mamba2)
        .input_dim(3)
        .output_dim(3)
        .hidden_dim(64)
        .state_dim(8)
        .num_layers(2)
        .build()?;

    println!(
        "Predictor ready  (context window = {})",
        predictor.context_window()
    );

    // Single-step prediction — O(1) complexity thanks to SSM recurrence
    let input = array![0.1f32, 0.2, 0.3];
    let output = predictor.step(&input)?;
    println!("Single step:  input {:?}  ->  output {:?}", input, output);

    // Multi-step autoregressive prediction (5 steps ahead)
    predictor.reset();
    let trajectory = predictor.predict_n(&input, 5)?;
    println!("Multi-step prediction (5 steps):");
    for (i, row) in trajectory.outer_iter().enumerate() {
        println!("  step {}: {:?}", i + 1, row);
    }

    // Batch prediction over a set of independent inputs
    predictor.reset();
    let batch_inputs = vec![
        array![0.1f32, 0.2, 0.3],
        array![0.4f32, 0.5, 0.6],
        array![0.7f32, 0.8, 0.9],
    ];
    let batch_outputs = predictor.predict_batch(&batch_inputs)?;
    println!("Batch prediction ({} inputs):", batch_inputs.len());
    for (i, (inp, out)) in batch_inputs.iter().zip(batch_outputs.iter()).enumerate() {
        println!("  input {}: {:?}  ->  {:?}", i + 1, inp, out);
    }

    // Forking — useful for exploring multiple prediction branches independently
    let _forked = predictor.fork()?;
    println!("Fork created (independent predictor with same state)");

    Ok(())
}

/// Demonstrates applying guardrails (constraint enforcement) to predictions.
fn example_constraints() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Constraint Enforcement ---");

    // Robotics preset: 3-DOF joint angles, input/output in radians
    let mut predictor = KizzasiBuilder::robotics_preset(3).build()?;

    // Run a prediction without constraints first
    let input = array![0.5f32, 1.0, -0.5];
    let unconstrained = predictor.step(&input)?;
    println!("Unconstrained output: {:?}", unconstrained);

    // Install guardrails: all joint angles must stay within [-pi, pi]
    let mut guardrails = GuardrailSet::new();
    let joint_constraint = ConstraintBuilder::new()
        .name("joint_limits")
        .greater_eq(-std::f32::consts::PI)
        .less_eq(std::f32::consts::PI)
        .build()?;
    guardrails.add_global(Guardrail::new(joint_constraint, false));
    predictor.set_guardrails(guardrails);

    println!("Guardrails installed: {}", predictor.has_guardrails());

    // Predictions are now projected into the safe region automatically
    predictor.reset();
    let constrained = predictor.step(&input)?;
    println!("Constrained output:   {:?}", constrained);

    // Validate specific values and measure violation severity
    let test_values = [
        array![0.0f32, 0.0, 0.0],  // trivially safe
        array![3.0f32, -3.0, 1.5], // just inside bounds
        array![4.0f32, 2.0, 4.0],  // outside bounds -> violation
    ];

    println!("Validation check:");
    for (i, val) in test_values.iter().enumerate() {
        let valid = predictor.validate(val);
        let loss = predictor.violation_loss(val);
        println!(
            "  value {}: {:?}  valid={}  violation_loss={:.4}",
            i + 1,
            val,
            valid,
            loss
        );
    }

    // Remove guardrails and confirm
    predictor.clear_guardrails();
    println!("After clearing guardrails: {}", predictor.has_guardrails());

    Ok(())
}
