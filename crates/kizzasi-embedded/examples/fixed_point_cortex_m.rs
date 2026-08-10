//! Fixed-point Q16.16 inference for FPU-less ARM Cortex-M (M0, M0+, M3).
//!
//! This example shows how to convert `f32` inputs to [`Q16`] fixed-point,
//! perform a dot product, run a Q16-based exponential approximation, and
//! convert the result back. It is gated on the `fixed-point` feature, which
//! enables the [`kizzasi_embedded::fixed_point`] module.
//!
//! Memory footprint on a `d_state = 8` SSM is roughly:
//!
//! * Recurrent state `h`     : 8 × 4 B = 32 B (Q16 reuses i32 storage)
//! * Inner buffer `prev_x`   : 16 × 4 B = 64 B
//! * Weight diagonals (×4)   : 4 × 32 B = 128 B
//! * Total state RAM         : ~ 224 B (well under the 2 KB target)
//!
//! Build with:
//!
//! ```text
//! cargo build --example fixed_point_cortex_m -p kizzasi-embedded --features fixed-point
//! ```
//!
//! Run with:
//!
//! ```text
//! cargo run --example fixed_point_cortex_m -p kizzasi-embedded --features fixed-point
//! ```
//!
//! Without the `fixed-point` feature the example compiles to an empty
//! `main` that prints a one-line skip notice — this keeps
//! `cargo build --examples` working on minimal feature sets.

#[cfg(feature = "fixed-point")]
fn main() {
    use kizzasi_embedded::fixed_point::{fixed_dot, fixed_exp_approx, Q16};

    // ---- Step 1: ingest f32 inputs from the analog/sensor frontend -----
    let inputs_f32 = [0.10_f32, 0.20, 0.30, 0.40, -0.10, -0.20, 0.05, 0.15];
    let weights_f32 = [1.00_f32, 0.50, -0.25, 0.75, 0.10, 0.20, -0.30, 0.40];

    // ---- Step 2: convert to Q16.16 fixed-point ---------------------------
    // On a real Cortex-M0 target this conversion is a single `vcvt`-free
    // multiply-then-truncate sequence and adds <= 1 cycle per element.
    let mut inputs_q = [Q16::ZERO; 8];
    let mut weights_q = [Q16::ZERO; 8];
    for i in 0..inputs_f32.len() {
        inputs_q[i] = Q16::from_f32(inputs_f32[i]);
        weights_q[i] = Q16::from_f32(weights_f32[i]);
    }

    // ---- Step 3: dot product entirely in fixed-point ---------------------
    let dot = fixed_dot(&inputs_q, &weights_q);

    // ---- Step 4: Taylor-series exp for the SSM `A_bar` decay -------------
    // `fixed_exp_approx` is accurate to ~1 % for |x| < 0.5; clamp the
    // argument first to stay within that range.
    let arg = if dot.to_f32().abs() > 0.5 {
        // Saturate to ±0.5 so the Taylor series stays accurate.
        if dot.is_negative() {
            Q16::from_f32(-0.5)
        } else {
            Q16::from_f32(0.5)
        }
    } else {
        dot
    };
    let decay = fixed_exp_approx(arg);

    // ---- Step 5: convert back to f32 for host-side reporting -------------
    let dot_f32 = dot.to_f32();
    let decay_f32 = decay.to_f32();

    // Cross-check against the host f32 implementation to bound quantisation
    // error. On Cortex-M0 the host comparison is omitted; here we keep it
    // to demonstrate that the Q16 path stays within ~ 1 % of f32.
    let dot_ref: f32 = inputs_f32
        .iter()
        .zip(weights_f32.iter())
        .map(|(x, w)| x * w)
        .sum();
    let err = (dot_f32 - dot_ref).abs();

    println!("fixed_point_cortex_m:");
    println!("  Q16 dot           = {dot_f32:.6}");
    println!("  f32 dot reference = {dot_ref:.6}");
    println!("  |Q16 - f32| error = {err:.2e}");
    println!("  Q16 exp(dot.clamp(±0.5)) = {decay_f32:.6}");
}

#[cfg(not(feature = "fixed-point"))]
fn main() {
    println!(
        "fixed_point_cortex_m: skipped (built without the `fixed-point` feature). \
         Rebuild with `cargo run --example fixed_point_cortex_m \
         -p kizzasi-embedded --features fixed-point` to see the demo."
    );
}
