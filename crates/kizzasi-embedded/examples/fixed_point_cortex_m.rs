//! Fixed-point Q16.16 inference for FPU-less ARM Cortex-M (M0, M0+, M3).
//!
//! This example runs a complete Mamba selective-SSM recurrence
//! ([`MambaStepQ16`](kizzasi_embedded::MambaStepQ16)) entirely in Q16.16
//! integer arithmetic — no `f32` operation is executed inside the loop, so an
//! M0+ never touches its soft-float library — and cross-checks it against the
//! `f32` kernel. It is gated on the `fixed-point` feature.
//!
//! Memory footprint of the `d_state = 8` configuration below (all `Q16` is a
//! single `i32` word, and none of it is heap-allocated):
//!
//! * Recurrent state `h`          : 8 × 4 B  =  32 B
//! * Weight diagonals `a_log`/`B`/`C` : 3 × 8 × 4 B  =  96 B
//! * Input frame `x`              : 8 × 4 B  =  32 B
//! * Total                        : 160 B, all in `.bss`/stack
//!
//! (The `f32` [`SsmState`](kizzasi_embedded::SsmState) also carries a
//! `d_inner`-long `prev_x` convolution history — `d_inner * 4` bytes, e.g.
//! 256 B at the `rp2040` preset. The Q16 path has no convolution stage, so it
//! carries only `h`.)
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
    use kizzasi_embedded::math::ln_approx;
    use kizzasi_embedded::{MambaStep, MambaStepQ16};

    const D_STATE: usize = 8;

    // ---- Step 1: ingest f32 inputs from the analog/sensor frontend -----
    let x_f32 = [0.10_f32, 0.20, 0.30, 0.40, -0.10, -0.20, 0.05, 0.15];
    // `a_log` is log-space: the effective state matrix is `A = -exp(a_log)`,
    // so checkpoint values are non-negative (HiPPO: ln(1), ln(2), ...).
    let a_log_f32: [f32; D_STATE] = core::array::from_fn(|n| ln_approx((n + 1) as f32));
    let b_f32 = [0.50_f32, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15];
    let c_f32 = [1.00_f32, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30];
    let delta_f32 = 0.10_f32;

    // ---- Step 2: convert to Q16.16 fixed-point ---------------------------
    // On a real Cortex-M0 target this conversion happens once, at model load,
    // and the hot loop below never sees an `f32` again.
    let mut x_q = [Q16::ZERO; D_STATE];
    let mut a_log_q = [Q16::ZERO; D_STATE];
    let mut b_q = [Q16::ZERO; D_STATE];
    let mut c_q = [Q16::ZERO; D_STATE];
    for i in 0..D_STATE {
        x_q[i] = Q16::from_f32(x_f32[i]);
        a_log_q[i] = Q16::from_f32(a_log_f32[i]);
        b_q[i] = Q16::from_f32(b_f32[i]);
        c_q[i] = Q16::from_f32(c_f32[i]);
    }
    let delta_q = Q16::from_f32(delta_f32);

    // ---- Step 3: run the recurrence in both precisions --------------------
    // The Q16 state is a plain array: no allocator is involved anywhere.
    let mut h_q = [Q16::ZERO; D_STATE];
    let mut h_f32 = [0.0_f32; D_STATE];

    let mut worst_err = 0.0_f32;
    let mut last = (0.0_f32, 0.0_f32);
    for _ in 0..32 {
        let y_q =
            MambaStepQ16::step_slice(&mut h_q, &x_q, &a_log_q, &b_q, &c_q, delta_q, Q16::ZERO)
                .expect("dimensions match");
        let y_f = MambaStep::step_slice(
            &mut h_f32, &x_f32, &a_log_f32, &b_f32, &c_f32, delta_f32, 0.0,
        )
        .expect("dimensions match");
        worst_err = worst_err.max((y_q.to_f32() - y_f).abs());
        last = (y_q.to_f32(), y_f);
    }

    // ---- Step 4: standalone Q16 primitives --------------------------------
    let dot = fixed_dot(&x_q, &b_q).expect("equal lengths");
    // `fixed_exp_approx` now range-reduces properly, so it is safe over the
    // whole Q16.16 range and never rises above 1 for a negative argument —
    // no call-site clamp required.
    let decay = fixed_exp_approx(dot.saturating_neg());

    println!("fixed_point_cortex_m ({D_STATE}-state Mamba, 32 steps):");
    println!("  Q16 y (last step)      = {:.6}", last.0);
    println!("  f32 y (last step)      = {:.6}", last.1);
    println!("  worst |Q16 - f32|      = {worst_err:.3e}");
    println!("  Q16 dot(x, B)          = {:.6}", dot.to_f32());
    println!("  Q16 exp(-dot)          = {:.6}", decay.to_f32());
    println!("  state RAM (Q16 h)      = {} B", D_STATE * 4);
}

#[cfg(not(feature = "fixed-point"))]
fn main() {
    println!(
        "fixed_point_cortex_m: skipped (built without the `fixed-point` feature). \
         Rebuild with `cargo run --example fixed_point_cortex_m \
         -p kizzasi-embedded --features fixed-point` to see the demo."
    );
}
