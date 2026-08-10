//! Minimal Mamba SSM step demonstrating the no_std-compatible API.
//!
//! This example builds and runs on a hosted (std) target for ease of demo,
//! but the underlying [`kizzasi_embedded`] code paths it exercises are all
//! `no_std` compatible: only the example's `main` and `println!` require
//! `std`.
//!
//! On a bare-metal target you would replace `main` with the appropriate
//! entry-point (`#[entry]` from `cortex-m-rt`, `_start` from
//! `riscv-rt`, etc.) and route the output via `defmt`, a UART, or a debug
//! register instead of `println!`.
//!
//! Build with:
//!
//! ```text
//! cargo build --example no_std_basic -p kizzasi-embedded
//! ```
//!
//! Run with:
//!
//! ```text
//! cargo run --example no_std_basic -p kizzasi-embedded
//! ```

use kizzasi_embedded::{MambaStep, SsmConfig, SsmState};

fn main() {
    // Build a tiny config: 8 input channels, 4-element recurrent state.
    // `SsmConfig::new` validates that all dimensions are non-zero, returning
    // `EmbeddedResult` rather than panicking.
    let config = SsmConfig::new(8, 4, 2).expect("dimensions are non-zero");

    // Allocate zeroed recurrent state. On a hosted target this uses the
    // global allocator; on a no_std target with the `alloc` feature it uses
    // whatever heap allocator the crate has been linked against
    // (e.g. `embedded-alloc`).
    let mut state = SsmState::new(&config);

    // Synthetic per-channel input — in a real application this would be one
    // frame of sensor / audio / language-model features.
    let x = [0.10_f32, 0.20, 0.30, 0.40];

    // Static SSM parameters: in a real deployment these come from flashed
    // model weights, here we hand-craft values that produce a stable
    // recurrence (negative `a_log`, modest `b`/`c`, small `delta`).
    let a_log = [-1.0_f32, -1.5, -2.0, -0.5];
    let b = [0.50_f32, 0.40, 0.30, 0.60];
    let c = [1.00_f32, 0.80, 1.20, 0.90];
    let delta = 0.10_f32;
    let d_skip = 0.05_f32;

    // Single recurrence step. After this call `state.h` carries the updated
    // SSM hidden state for the next time step.
    let y = MambaStep::step(&mut state, &x, &a_log, &b, &c, delta, d_skip)
        .expect("inputs are correctly dimensioned");

    println!(
        "no_std_basic: y = {y:.6}, state.h = [{:.4}, {:.4}, {:.4}, {:.4}]",
        state.h[0], state.h[1], state.h[2], state.h[3]
    );
}
