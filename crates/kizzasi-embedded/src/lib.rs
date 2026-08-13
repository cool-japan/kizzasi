//! Kizzasi Embedded — no_std SSM inference for edge/embedded devices
//!
//! This crate implements lightweight SSM (State Space Model) inference
//! suitable for ARM Cortex-M, RISC-V, and other embedded targets.
//!
//! # Features
//! - `std` (default): enables the standard library; implies `alloc` and
//!   provides `f32` math methods (`sqrt`, `round`, …) via libstd.
//! - `alloc`: enables heap allocation via the `alloc` crate
//!   (`Vec`, `Box`, …). Required by the owned state types
//!   ([`SsmState`], [`S4State`]) and the batch quantisation helpers. The
//!   `*_slice` kernels work without it.
//! - `libm`: provides `f32` math shims (`sqrt`, `round`, …) for
//!   `no_std` targets. Required when `std` is disabled.
//! - `fixed-point`: enables Q16.16 fixed-point arithmetic in
//!   the `fixed_point` module **and** the fully integer SSM recurrence in
//!   the `ssm_fixed` module; useful for Cortex-M0/M0+ and other FPU-less cores.
//!
//! # Build matrix
//!
//! | Features                               | What you get                                     |
//! |----------------------------------------|--------------------------------------------------|
//! | `std` (default)                        | Full hosted build: alloc + std f32 math          |
//! | `alloc, libm`                          | Bare-metal `no_std` build with `libm` math       |
//! | `libm` only (no `alloc`)               | Bare-metal build with **no allocator**: the      |
//! |                                        | `*_slice` kernels over caller-owned arrays       |
//! | `alloc, libm, fixed-point`             | Bare-metal build with the Q16.16 SSM added       |
//! | `alloc` only (no `std`, no `libm`)     | Compile error: f32 math not available            |
//!
//! For bare-metal targets without `std`, use
//! `--no-default-features --features alloc,libm`, or drop `alloc` entirely if
//! the firmware has no allocator.
//!
//! # Usage without an allocator
//!
//! Every kernel has an allocator-free `*_slice` entry point that borrows the
//! recurrent state, so the state can live in a plain array in `.bss`:
//!
//! ```
//! use kizzasi_embedded::MambaStep;
//!
//! let mut h = [0.0_f32; 4];
//! // Checkpoint convention: A = -exp(a_log), so `a_log` is log-space and
//! // real weights are non-negative.
//! let a_log = [0.0_f32, 0.693_147, 1.098_612, 1.386_294];
//! let x = [0.1_f32, 0.2, 0.3, 0.4];
//! let b = [0.5_f32; 4];
//! let c = [1.0_f32; 4];
//!
//! let y = MambaStep::step_slice(&mut h, &x, &a_log, &b, &c, 0.1, 0.05)
//!     .expect("all slices have length 4");
//! assert!(y.is_finite());
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

// The crate relies on `f32` math (sqrt, round, …) in `math.rs` and
// `quantize.rs`. When neither `std` nor `libm` is enabled there is no way to
// provide those operations, so emit a clear compile-time error rather than
// silently producing wrong results.
#[cfg(all(not(feature = "std"), not(feature = "libm")))]
compile_error!(
    "kizzasi-embedded requires either the `std` (default) or `libm` feature \
     to provide f32 math operations. For bare-metal targets, use \
     `--no-default-features --features alloc,libm`."
);

// `Vec` and the `vec!` macro live in the `alloc` crate when we build
// without `std`. The `std` feature implies `alloc`, so this `extern crate`
// covers both paths.
#[cfg(feature = "alloc")]
extern crate alloc;

pub mod error;
#[cfg(feature = "fixed-point")]
pub mod fixed_point;
pub mod math;
pub mod presets;
pub mod quantize;
pub mod ssm;
#[cfg(feature = "fixed-point")]
pub mod ssm_fixed;

pub use error::{EmbeddedError, EmbeddedResult};
pub use presets::{esp32c3, rp2040, stm32h7};
pub use ssm::{DepthwiseConv1d, MambaStep, S4Step, SsmConfig};

#[cfg(feature = "alloc")]
pub use ssm::{S4State, SsmState};

#[cfg(feature = "fixed-point")]
pub use ssm_fixed::MambaStepQ16;

#[cfg(all(feature = "fixed-point", feature = "alloc"))]
pub use ssm_fixed::Q16SsmState;
