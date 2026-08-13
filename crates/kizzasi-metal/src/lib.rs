//! Apple Metal backend activation for the Kizzasi ecosystem.
//!
//! This crate exists to solve a Cargo problem, not a Rust one. `kizzasi-core`
//! exposes a `metal` feature that is meaningful only on Apple hardware, but a
//! Cargo *feature* cannot be made conditional on the target: writing
//! `metal = ["candle-core/metal"]` enables candle's Metal backend on Linux and
//! Windows too, and that backend depends on `objc2`, whose `lib.rs` opens with
//! a `compile_error!` on non-Apple platforms. The practical fallout was that
//! `cargo build --all-features` — the command the README documents and CI runs
//! — could not complete anywhere except macOS.
//!
//! A Cargo *dependency*, unlike a feature, can be target-scoped, and under
//! `resolver = "2"` the features requested by a platform-specific dependency
//! are ignored for targets that are not currently being built. So this crate
//! declares `candle-core` twice — once behind `cfg(target_vendor = "apple")`
//! with `features = ["metal"]`, once behind `cfg(not(target_vendor = "apple"))`
//! without — and `kizzasi-core`'s `metal` feature simply pulls this crate in.
//! Apple builds light up the real GPU backend; everywhere else the feature is
//! inert and the build succeeds.
//!
//! The alternative — a renamed `package = "candle-core"` alias scoped to the
//! Apple target inside `kizzasi-core` itself — was tried first and rejected by
//! Cargo: `cargo metadata --filter-platform` refuses a manifest that "depends
//! on crate candle-core multiple times with different names", which breaks
//! `cargo-nextest` and `cargo-deny`. Moving the second declaration into its own
//! package keeps every manifest single-named and every tool happy.
//!
//! # What "inert" means
//!
//! It means an honest failure, never a fabricated success. On a non-Apple
//! target [`BACKEND_COMPILED`] is `false`, [`is_available`] returns `false`,
//! [`device_ordinals`] returns an empty slice, and [`new_device`] returns an
//! error naming the host target — it does not silently hand back a CPU device
//! that pretends to be a GPU.
//!
//! # Usage
//!
//! Reach for this crate through `kizzasi-core` rather than directly:
//!
//! ```toml
//! [dependencies]
//! kizzasi-core = { version = "0.2", features = ["metal"] }
//! ```
//!
//! ```rust
//! // Works on every platform; only Apple builds can return `Ok`.
//! match kizzasi_metal::new_device(0) {
//!     Ok(device) => println!("Metal device ready: {device:?}"),
//!     Err(err) => println!("no Metal device: {err}"),
//! }
//! ```
//!
//! # Purity
//!
//! NOTE: FFI. On Apple targets this crate transitively links Apple's Metal
//! framework through `objc2`. That is why the whole path stays opt-in behind
//! `kizzasi-core/metal`; the default Kizzasi build is pure Rust.

#![deny(missing_docs)]

use candle_core::{Device, Error as CandleError, Result as CandleResult};

/// Whether candle's Metal backend was actually compiled into this build.
///
/// This is a target property, not a feature property: the `metal` feature can
/// be enabled on a Linux or Windows build (`--all-features` does exactly that)
/// without the backend existing. Code that needs to know whether a Metal
/// device could ever be produced should consult this constant rather than
/// `cfg!(feature = "metal")`.
pub const BACKEND_COMPILED: bool = cfg!(target_vendor = "apple");

/// Human-readable reason why the Metal backend is unavailable, or `None` when
/// it is compiled in.
///
/// Used for error messages that would otherwise have to guess at the cause.
pub const fn unavailable_reason() -> Option<&'static str> {
    if BACKEND_COMPILED {
        None
    } else {
        Some(
            "candle's Metal backend is compiled only for Apple targets \
             (`cfg(target_vendor = \"apple\")`); this binary was built for a \
             non-Apple target",
        )
    }
}

/// Open Metal device `ordinal`.
///
/// On Apple targets this forwards to [`Device::new_metal`], so it fails when
/// the machine has no Metal-capable GPU or the ordinal is out of range. On
/// every other target it fails immediately with the reason from
/// [`unavailable_reason`].
pub fn new_device(ordinal: usize) -> CandleResult<Device> {
    match unavailable_reason() {
        None => Device::new_metal(ordinal),
        Some(reason) => Err(CandleError::Msg(format!(
            "cannot open Metal device {ordinal}: {reason}"
        ))),
    }
}

/// Whether a Metal device can be opened right now.
///
/// Returns `false` both when the backend is not compiled in and when it is but
/// no device answers — the caller usually wants to fall back to CPU either way.
pub fn is_available() -> bool {
    BACKEND_COMPILED && new_device(0).is_ok()
}

/// Ordinals of the Metal devices that can actually be opened.
///
/// Only ordinal 0 is probed. candle's Metal backend indexes an internal `Vec`
/// without bounds-checking on some paths, so walking ordinals upward can panic
/// rather than return an error on multi-GPU Macs; a single probe is the
/// conservative behaviour that the rest of Kizzasi has always relied on.
pub fn device_ordinals() -> Vec<usize> {
    if is_available() {
        vec![0]
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_compiled_matches_target() {
        assert_eq!(BACKEND_COMPILED, cfg!(target_vendor = "apple"));
        assert_eq!(unavailable_reason().is_none(), BACKEND_COMPILED);
    }

    #[test]
    fn non_apple_targets_fail_loudly() {
        if BACKEND_COMPILED {
            return;
        }
        let err = new_device(0).expect_err("non-Apple builds must not open a Metal device");
        let message = err.to_string();
        assert!(
            message.contains("Apple"),
            "error must name the cause, got: {message}"
        );
        assert!(!is_available());
        assert!(device_ordinals().is_empty());
    }

    #[test]
    fn availability_agrees_with_ordinals() {
        assert_eq!(is_available(), !device_ordinals().is_empty());
    }
}
