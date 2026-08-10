//! STM32H7 preset usage: a 50 KB-weight Mamba block on 1 MB SRAM.
//!
//! Demonstrates the [`stm32h7`](kizzasi_embedded::stm32h7) preset together
//! with [`SsmState`] and [`MambaStep`]. The preset selects
//! `d_model = 64`, `d_state = 16`, `expand = 2`, which is the most
//! capable configuration that still leaves plenty of headroom for
//! activations, DMA scratch, and the IDF / HAL runtime on a 1 MB
//! STM32H743-class device.
//!
//! Build with:
//!
//! ```text
//! cargo build --example stm32h7_preset -p kizzasi-embedded
//! ```
//!
//! Run with:
//!
//! ```text
//! cargo run --example stm32h7_preset -p kizzasi-embedded
//! ```

use kizzasi_embedded::{stm32h7, MambaStep, SsmState};

fn main() {
    // Select the STM32H7 preset.
    let config = stm32h7();

    // Heap-allocated zero-initialised state. On bare-metal STM32H7 this
    // would back onto `embedded-alloc`'s `CortexMHeap`, but in this hosted
    // demo we just use the global allocator.
    let mut state = SsmState::new(&config);

    // Synthetic SSM diagonals sized to the preset's `d_state = 16`.
    let d_state = config.d_state;
    let x: alloc::vec::Vec<f32> = (0..d_state).map(|i| 0.01_f32 * i as f32).collect();
    let a_log: alloc::vec::Vec<f32> = (0..d_state).map(|i| -0.5 - 0.1 * i as f32).collect();
    let b: alloc::vec::Vec<f32> = (0..d_state).map(|_| 0.30_f32).collect();
    let c: alloc::vec::Vec<f32> = (0..d_state).map(|i| 0.5 + 0.05 * i as f32).collect();

    // Run a handful of recurrence steps to demonstrate state evolution.
    let delta = 0.05_f32;
    let d_skip = 0.0_f32;
    let mut history = [0.0_f32; 4];
    for (i, slot) in history.iter_mut().enumerate() {
        let y = MambaStep::step(&mut state, &x, &a_log, &b, &c, delta, d_skip)
            .expect("preset dimensions match step inputs");
        *slot = y;
        // Slowly perturb the input so the state actually moves between steps.
        // (No-op on `x` here; the recurrence itself is the visible change.)
        let _ = i;
    }

    println!(
        "stm32h7_preset: d_model={}, d_state={}, d_inner={}",
        config.d_model, config.d_state, config.d_inner
    );
    println!(
        "  outputs over 4 steps = [{:.6}, {:.6}, {:.6}, {:.6}]",
        history[0], history[1], history[2], history[3]
    );
    println!(
        "  final state.h[0..4]  = [{:.6}, {:.6}, {:.6}, {:.6}]",
        state.h[0], state.h[1], state.h[2], state.h[3]
    );
}

// Pull in the `alloc` crate via the hosted `std` so the example can use
// `Vec` without enabling extra features. On a true no_std target the user
// would write `extern crate alloc;` instead and link against an allocator.
extern crate alloc;
