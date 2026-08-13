//! Platform-specific SSM presets.
//!
//! Each preset returns an [`SsmConfig`] tuned for a specific embedded
//! target's flash/RAM budget. The numbers come from common datasheets:
//!
//! * **STM32H7** — 2 MB flash, up to 1 MB SRAM, Cortex-M7 with FPU.
//! * **RP2040**  — 264 KB SRAM, dual Cortex-M0+ (no FPU), external XIP flash.
//! * **ESP32-C3** — 400 KB SRAM, single-core RV32IMC (no FPU), 4 MB external flash.
//!
//! The presets pick `d_model` / `d_state` / `expand` values so that the
//! recurrent state and the on-stack working buffers stay well within the
//! target's RAM budget.
//!
//! # Examples
//!
//! ```
//! use kizzasi_embedded::{stm32h7, SsmState};
//!
//! let cfg = stm32h7();
//! let state = SsmState::new(&cfg);
//! assert_eq!(state.h.len(), cfg.d_state);
//! ```
//!
//! All presets are pure functions and may be evaluated at compile time via
//! `const fn` wrappers in future revisions; today they return a freshly
//! constructed [`SsmConfig`] each call.
//!
//! # State memory
//!
//! The figures below are **derived from the code**, not estimated:
//! [`SsmState`](crate::SsmState) allocates `d_state` floats for `h` and
//! `d_inner` floats for `prev_x`, so its footprint is exactly
//! `(d_state + d_inner) * 4` bytes. The test
//! `presets::tests::test_documented_state_ram_matches_allocation` keeps this
//! table honest.
//!
//! | Preset      | `d_model` | `d_state` | `d_inner` | `SsmState` (f32) | `Q16SsmState` |
//! |-------------|-----------|-----------|-----------|------------------|---------------|
//! | `stm32h7`   | 64        | 16        | 128       | 576 B            | 64 B          |
//! | `rp2040`    | 32        | 8         | 64        | 288 B            | 32 B          |
//! | `esp32c3`   | 48        | 12        | 96        | 432 B            | 48 B          |
//!
//! The last column is the Q16.16 state used by
//! `ssm_fixed::MambaStepQ16` (feature `fixed-point`), which keeps only `h`
//! (`d_state * 4` bytes) because the fixed-point path has no convolution
//! history. Transient activation buffers are caller-supplied and not counted,
//! and the `*_slice` kernels let the caller place both buffers in `.bss`
//! instead of on a heap.
//!
//! **Weight sizes are deliberately not quoted.** This crate contains no
//! projection weights, no weight-loading code and no block structure — only
//! the diagonal recurrence — so any "KB of INT8 weights per block" figure
//! here would describe a model this crate cannot see. Size your flash budget
//! from the checkpoint you actually deploy.

use crate::ssm::SsmConfig;

/// STM32H7 preset: `d_model = 64`, `d_state = 16`, `expand = 2`.
///
/// Targets the high-end Cortex-M7 family (e.g. STM32H743, STM32H753) with
/// 1 MB SRAM and a single-precision FPU. The recurrent state costs 576 B
/// (`(16 + 128) * 4`), leaving ample headroom for activations and per-frame
/// audio/sensor buffers.
///
/// # Rationale
///
/// * `d_model = 64` matches the natural width of many sensor-feature
///   extractors (e.g. 64-bin mel spectrograms).
/// * `d_state = 16` gives ample temporal capacity for sequences up to
///   ~256 steps without state saturation.
/// * `expand = 2` follows the canonical Mamba ratio, yielding `d_inner = 128`.
#[must_use]
pub fn stm32h7() -> SsmConfig {
    SsmConfig {
        d_model: 64,
        d_state: 16,
        d_inner: 128,
    }
}

/// RP2040 preset: `d_model = 32`, `d_state = 8`, `expand = 2`.
///
/// Targets the dual Cortex-M0+ on the Raspberry Pi Pico / RP2040 SoC. The
/// M0+ lacks an FPU, so callers should enable the `fixed-point` feature and
/// run `ssm_fixed::MambaStepQ16`, which executes the
/// entire recurrence in `fixed_point::Q16` integer arithmetic —
/// 32 bytes of state at this size, and no soft-float calls in the hot loop.
///
/// The `f32` [`SsmState`](crate::SsmState) equivalent is 288 bytes.
///
/// # Rationale
///
/// * `d_model = 32` is the smallest size that retains useful expressivity
///   for binary classifiers and small-vocabulary keyword spotters.
/// * `d_state = 8` keeps the inner loop iteration count low enough to
///   compile to a tight, branch-free sequence on M0+.
/// * `expand = 2` yields `d_inner = 64`, so the convolution history `prev_x`
///   is 256 bytes — four 64-byte lines on targets that have a cache at all.
#[must_use]
pub fn rp2040() -> SsmConfig {
    SsmConfig {
        d_model: 32,
        d_state: 8,
        d_inner: 64,
    }
}

/// ESP32-C3 preset: `d_model = 48`, `d_state = 12`, `expand = 2`.
///
/// Targets the single-core RISC-V (RV32IMC) ESP32-C3 with 400 KB SRAM and
/// no hardware FPU. The default soft-float path in `core` is acceptable for
/// this preset, though the `fixed-point` feature is recommended where every
/// microsecond counts (e.g. real-time audio).
///
/// The recurrent state costs 432 B (`(12 + 96) * 4`), comfortably fitting in
/// the ~300 KB usable SRAM after subtracting the IDF runtime (~64 KB BSS) and
/// Wi-Fi/BLE buffers (~32 KB).
///
/// # Rationale
///
/// * `d_model = 48` (3·16) aligns naturally with the SIMD-less but
///   loop-unrollable RV32IMC instruction set.
/// * `d_state = 12` provides more recurrent capacity than the RP2040 preset
///   without crossing the 16-element threshold beyond which loop overhead
///   begins to dominate on this core.
/// * `expand = 2` gives `d_inner = 96`, a multiple of both 16 and 24 to
///   admit cache-friendly tiling strategies.
#[must_use]
pub fn esp32c3() -> SsmConfig {
    SsmConfig {
        d_model: 48,
        d_state: 12,
        d_inner: 96,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stm32h7_dimensions() {
        let cfg = stm32h7();
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.d_state, 16);
        assert_eq!(cfg.d_inner, 128);
    }

    #[test]
    fn test_rp2040_dimensions() {
        let cfg = rp2040();
        assert_eq!(cfg.d_model, 32);
        assert_eq!(cfg.d_state, 8);
        assert_eq!(cfg.d_inner, 64);
    }

    #[test]
    fn test_esp32c3_dimensions() {
        let cfg = esp32c3();
        assert_eq!(cfg.d_model, 48);
        assert_eq!(cfg.d_state, 12);
        assert_eq!(cfg.d_inner, 96);
    }

    #[test]
    fn test_presets_are_nonzero() {
        for cfg in [stm32h7(), rp2040(), esp32c3()] {
            assert!(cfg.d_model > 0, "preset d_model must be > 0");
            assert!(cfg.d_state > 0, "preset d_state must be > 0");
            assert!(cfg.d_inner > 0, "preset d_inner must be > 0");
        }
    }

    #[test]
    fn test_presets_expand_consistent() {
        // All presets follow the canonical Mamba `expand = 2` ratio.
        for cfg in [stm32h7(), rp2040(), esp32c3()] {
            assert_eq!(
                cfg.d_inner,
                cfg.d_model * 2,
                "preset d_inner must equal d_model * 2 (canonical Mamba expand=2)"
            );
        }
    }

    #[test]
    fn test_documented_state_ram_matches_allocation() {
        // Keeps the module-level memory table tied to the code that
        // implements it, instead of to a hand-written estimate.
        const F32_BYTES: usize = core::mem::size_of::<f32>();
        for (name, cfg, documented_f32, documented_q16) in [
            ("stm32h7", stm32h7(), 576_usize, 64_usize),
            ("rp2040", rp2040(), 288, 32),
            ("esp32c3", esp32c3(), 432, 48),
        ] {
            let state = crate::ssm::SsmState::new(&cfg);
            let actual = (state.h.len() + state.prev_x.len()) * F32_BYTES;
            assert_eq!(
                actual, documented_f32,
                "{name}: documented SsmState size {documented_f32} B, actual {actual} B"
            );
            let q16 = cfg.d_state * core::mem::size_of::<i32>();
            assert_eq!(
                q16, documented_q16,
                "{name}: documented Q16SsmState size {documented_q16} B, actual {q16} B"
            );
        }
    }

    #[test]
    fn test_presets_pass_validation() {
        // The presets are struct literals, so `SsmConfig::new`'s checks are
        // bypassed; `validate` is what re-establishes the invariant.
        for cfg in [stm32h7(), rp2040(), esp32c3()] {
            assert!(cfg.validate().is_ok(), "preset must pass validate()");
        }
    }

    #[test]
    fn test_presets_ordering_by_capacity() {
        // RP2040 should be the smallest, STM32H7 the largest.
        let rp = rp2040();
        let esp = esp32c3();
        let stm = stm32h7();
        assert!(
            rp.d_model < esp.d_model && esp.d_model < stm.d_model,
            "expected RP2040 < ESP32-C3 < STM32H7 by d_model, got {} < {} < {}",
            rp.d_model,
            esp.d_model,
            stm.d_model
        );
        assert!(
            rp.d_state < esp.d_state && esp.d_state < stm.d_state,
            "expected RP2040 < ESP32-C3 < STM32H7 by d_state"
        );
    }
}
