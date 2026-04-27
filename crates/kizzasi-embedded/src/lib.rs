//! Kizzasi Embedded — no_std SSM inference for edge/embedded devices
//!
//! This crate implements lightweight SSM (State Space Model) inference
//! suitable for ARM Cortex-M, RISC-V, and other embedded targets.
//!
//! # Features
//! - `std` (default): enables standard library
//! - `fixed-point`: enables fixed-point Q16.16 arithmetic
//! - `alloc`: use alloc crate (enabled automatically when std is disabled)
//!
//! # Usage (no_std)
//! ```no_run
//! // In Cargo.toml: kizzasi-embedded = { default-features = false, features = ["alloc"] }
//! extern crate alloc;
//! use kizzasi_embedded::ssm::{SsmConfig, SsmState};
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod error;
#[cfg(feature = "fixed-point")]
pub mod fixed_point;
pub mod math;
pub mod quantize;
pub mod ssm;

pub use error::{EmbeddedError, EmbeddedResult};
pub use ssm::{MambaStep, S4Step, SsmConfig, SsmState};
