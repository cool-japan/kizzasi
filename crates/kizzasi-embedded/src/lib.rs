//! Kizzasi Embedded — no_std SSM inference for edge/embedded devices
//!
//! This crate implements lightweight SSM (State Space Model) inference
//! suitable for ARM Cortex-M, RISC-V, and other embedded targets.
//!
//! # Features
//! - `std` (default): enables the standard library; implies `alloc` and
//!   provides `f32` math methods (`sqrt`, `round`, …) via libstd.
//! - `alloc`: enables heap allocation via the `alloc` crate
//!   (`Vec`, `Box`, …). Required by the recurrent state vectors and the
//!   batch quantisation helpers.
//! - `libm`: provides `f32` math shims (`sqrt`, `round`, …) for
//!   `no_std` targets. Required when `std` is disabled.
//! - `fixed-point`: enables Q16.16 fixed-point arithmetic in
//!   [`fixed_point`]; useful for Cortex-M0/M0+ and other FPU-less cores.
//!
//! # Build matrix
//!
//! | Features                               | What you get                                   |
//! |----------------------------------------|------------------------------------------------|
//! | `std` (default)                        | Full hosted build: alloc + std f32 math        |
//! | `alloc, libm`                          | Bare-metal `no_std` build with `libm` math     |
//! | `alloc, libm, fixed-point`             | Bare-metal build with Q16.16 fixed-point added |
//! | `alloc` only (no `std`, no `libm`)     | Compile error: f32 math not available          |
//!
//! For bare-metal targets without `std`, use
//! `--no-default-features --features alloc,libm`.
//!
//! # Usage (`no_std`)
//! ```no_run
//! // In Cargo.toml:
//! //   kizzasi-embedded = { default-features = false, features = ["alloc", "libm"] }
//! extern crate alloc;
//! use kizzasi_embedded::ssm::{SsmConfig, SsmState};
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

pub use error::{EmbeddedError, EmbeddedResult};
pub use presets::{esp32c3, rp2040, stm32h7};
pub use ssm::{MambaStep, S4Step, SsmConfig, SsmState};
