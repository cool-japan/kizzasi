# kizzasi-embedded

![status](https://img.shields.io/badge/status-API%20evolving-yellow)
![version](https://img.shields.io/badge/version-0.2.3-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)
![published](https://img.shields.io/badge/crates.io-not%20published-lightgrey)

Embedded/edge inference for Kizzasi AGSP State Space Models on `no_std` targets.

> **Not on crates.io.** This crate is `publish = false`: it ships inside the
> [kizzasi repository](https://github.com/cool-japan/kizzasi) and is consumed
> as a path dependency (see [Installation](#installation)). Its API is still
> being shaped around real bare-metal targets, so it is deliberately kept out
> of the semver-stable release set.

## Overview

Bare-metal SSM inference engine with O(1) per-step complexity and optional Q16.16 fixed-point arithmetic for FPU-less cores. Designed for microcontrollers, FPGAs, and other resource-constrained devices.

Every kernel has two entry points:

- a **slice** form (`MambaStep::step_slice`, `S4Step::step_slice`, `DepthwiseConv1d::step_slice`, `MambaStepQ16::step_slice`) that borrows the recurrent state as `&mut [f32]` / `&mut [Q16]`. It needs **no allocator at all**, so the state can live in a plain array in `.bss`.
- a **state** form (`MambaStep::step`, …) taking the owned `SsmState` / `S4State` / `Q16SsmState` wrappers, which require the `alloc` feature.

Neither form allocates at inference time. A true bare-metal build needs only `libm` for `f32` math — no operating system, and no heap unless you want one.

## Features

- **`no_std` Compatible**: builds with `--no-default-features --features libm` (no allocator) or `alloc,libm` (with one)
- **O(1) Inference**: constant-time single-step recurrence (`MambaStep`, `S4Step`, `DepthwiseConv1d`) suitable for real-time loops
- **Fixed-Point SSM**: the `fixed-point` feature adds `Q16` (Q16.16) arithmetic **and `MambaStepQ16`**, which runs the entire selective-SSM recurrence — softplus, exp, ZOH discretisation, state update, output projection — in integer arithmetic
- **INT8 Quantization**: symmetric per-tensor `quantize`/`dequantize` helpers for compact weight storage
- **Platform Presets**: ready-made `SsmConfig`s for `stm32h7`, `rp2040`, and `esp32c3`, sized to each target's SRAM budget
- **Portable Math Kernels**: `exp`/`ln`/`softplus`/`sigmoid`/`silu`/`sin`/`cos`/`softmax`/`layer_norm`/`dot`/`matvec`, range-reduced polynomial approximations using only `core`, each with a measured accuracy contract in its rustdoc
- **Checked, Total API**: every fallible entry point returns `EmbeddedResult`; nothing panics on caller input, and `EmbeddedError` implements `core::error::Error`
- **Minimal Dependencies**: only `libm` as an optional dependency; no OS services required

## Feature Flags

| Feature | Default | Description |
|---|---|---|
| `std` | Yes | Standard library support; implies `alloc` and provides `f32` math (`sqrt`, `round`) via libstd |
| `alloc` | Yes (via `std`) | Heap allocation for the owned state types (`SsmState`, `S4State`, `Q16SsmState`) and the batch quantization helper. The `*_slice` kernels work without it |
| `libm` | No | Portable `f32` math shims for `no_std` targets; required when `std` is disabled |
| `fixed-point` | No | Q16.16 arithmetic (`fixed_point`) plus the fully integer SSM recurrence (`ssm_fixed`) |

`alloc` alone, without `std` or `libm`, fails to compile with a clear `compile_error!` — there is no source of `f32` math in that combination. For bare-metal builds use `--no-default-features --features alloc,libm`, or drop `alloc` if the firmware has no allocator. Add `,fixed-point` and call `MambaStepQ16` to keep the SSM recurrence itself off the FPU entirely.

## API Overview

Re-exported at the crate root: `SsmConfig`, `MambaStep`, `S4Step`, `DepthwiseConv1d`, `EmbeddedError`, `EmbeddedResult`, the presets `stm32h7` / `rp2040` / `esp32c3`, plus `SsmState` / `S4State` (feature `alloc`) and `MambaStepQ16` / `Q16SsmState` (feature `fixed-point`).

| Module | Contents |
|---|---|
| `ssm` | `SsmConfig` (`new` with overflow-checked `d_model * expand`, `validate`, `mamba_tiny`, `mamba_small`); `SsmState` (`h`, `prev_x`) and `S4State` (`h_re`, `h_im`); `MambaStep` (selective SSM, exact ZOH); `DepthwiseConv1d` (causal kernel-2 token shift over `d_inner`, the consumer of `prev_x`); `S4Step` (diagonal S4D with genuine complex poles) |
| `ssm_fixed` (feature `fixed-point`) | `MambaStepQ16`, `Q16SsmState` — the same Mamba recurrence entirely in Q16.16 integer arithmetic |
| `presets` | `stm32h7()`, `rp2040()`, `esp32c3()` — platform-tuned `SsmConfig` values |
| `fixed_point` (feature `fixed-point`) | `Q16` (saturating, round-to-nearest Q16.16), `fixed_dot`, `fixed_exp_approx`, `fixed_ln_approx`, `fixed_softplus`, `fixed_sigmoid`, `fixed_silu` |
| `quantize` | `compute_scale`, `quantize_scalar`, `dequantize_into`, `quantize_symmetric` — symmetric INT8 quantization |
| `math` | `exp_approx`, `ln_approx`, `softplus`, `sigmoid`, `silu`, `sin_approx`, `cos_approx`, `softmax_inplace`, `layer_norm`, `dot`, `matvec` |
| `error` | `EmbeddedError` (`DimensionMismatch`, `InvalidConfig`, `NumericalInstability`, `BufferTooSmall`), `EmbeddedResult<T>`, `impl core::error::Error` |

70 public items across seven modules, 0 stub markers (`todo!()`/`unimplemented!()`), 112 unit/integration tests plus 3 doc tests passing (`cargo nextest run -p kizzasi-embedded --all-features`).

### Parameter convention

`a_log` is **log-space**: the effective state matrix is `A = -exp(a_log)`, exactly as in `kizzasi-model` (`mixer.A_log` / `ssm.log_a`) and `kizzasi-core` (`mamba2.rs`). Real checkpoint values are therefore **non-negative** — the HiPPO initialisation is `a_log[n] = ln(n + 1)`. `A < 0` makes `A_bar = exp(delta * A)` land in `(0, 1)`, so the recurrence is unconditionally contracting.

### Accuracy contracts

Each approximation documents the worst error measured by its unit tests against the libstd reference:

| Function | Domain | Worst error |
|---|---|---|
| `exp_approx` | `[-88, 88]` | `< 1e-6` relative |
| `ln_approx` | `(0, 1e6]` | `< 1e-6` relative |
| `softplus` | `[-40, 40]` | `< 1e-5` relative |
| `sigmoid` | `[-40, 40]` | `< 1e-6` relative |
| `sin_approx` / `cos_approx` | `[-1e3, 1e3]` | `< 1e-5` absolute |
| `fixed_exp_approx` (Q16) | `[-12, 10]` | `<= 1e-3 * e^x + 3 LSB` |
| `fixed_ln_approx` (Q16) | `[1e-3, 1e4]` | `< 1e-4` absolute |
| `MambaStepQ16` vs `MambaStep` | `rp2040` preset, 256 steps | `< 2e-3` absolute |

## Installation

The crate is not published, so depend on it by path from inside the workspace:

```toml
# Cargo.toml
[dependencies]
kizzasi-embedded = { path = "crates/kizzasi-embedded" }

# ...or a bare-metal build with no allocator:
kizzasi-embedded = { path = "crates/kizzasi-embedded", default-features = false, features = ["libm", "fixed-point"] }
```

From outside the repository, use a git dependency pointing at the same path.

## Basic Usage

The example below runs on a hosted (`std`) target for convenience, but every `kizzasi_embedded` call it makes is `no_std`-compatible — only the surrounding `main`/`println!` need `std`. See `examples/no_std_basic.rs` for the full runnable version.

```rust
use kizzasi_embedded::{MambaStep, SsmConfig, SsmState};

// `SsmConfig::new(d_model, d_state, expand)` validates that every dimension
// is non-zero and that `d_model * expand` does not overflow, returning
// `EmbeddedResult` rather than panicking.
let config = SsmConfig::new(8, 4, 2).expect("dimensions are valid");
let mut state = SsmState::new(&config);

// One frame of input (length == d_state) plus the per-step SSM diagonals.
// `a_log` is log-space (A = -exp(a_log)), so these are the HiPPO values
// ln(1), ln(2), ln(3), ln(4).
let x = [0.10_f32, 0.20, 0.30, 0.40];
let a_log = [0.000_000_f32, 0.693_147, 1.098_612, 1.386_294];
let b = [0.50_f32, 0.40, 0.30, 0.60];
let c = [1.00_f32, 0.80, 1.20, 0.90];
let delta = 0.10_f32;
let d_skip = 0.05_f32;

// One recurrence step; `state.h` now holds the updated hidden state.
let y = MambaStep::step(&mut state, &x, &a_log, &b, &c, delta, d_skip)
    .expect("inputs are correctly dimensioned");

// The same kernel with no allocator at all — the state is a plain array.
let mut h = [0.0_f32; 4];
let y_slice = MambaStep::step_slice(&mut h, &x, &a_log, &b, &c, delta, d_skip)
    .expect("inputs are correctly dimensioned");
assert_eq!(y, y_slice);
```

## Fixed-Point Example

```rust
// Enable: kizzasi-embedded = { features = ["fixed-point"] }
use kizzasi_embedded::fixed_point::Q16;
use kizzasi_embedded::MambaStepQ16;

// Heap-free Q16.16 state: 4 words, 16 bytes.
let mut h = [Q16::ZERO; 4];
let x = [Q16::from_f32(0.25); 4];
let a_log = [
    Q16::ZERO,
    Q16::from_f32(0.693_147),
    Q16::from_f32(1.098_612),
    Q16::from_f32(1.386_294),
];
let b = [Q16::from_f32(0.5); 4];
let c = [Q16::from_f32(1.0); 4];

// Not a single floating-point instruction is executed inside this call.
let y = MambaStepQ16::step_slice(
    &mut h, &x, &a_log, &b, &c, Q16::from_f32(0.1), Q16::ZERO,
).expect("dimensions match");
assert!(y.to_f32().abs() < 1.0);
```

Every `Q16` operation **saturates** — `+`, `-`, `*`, unary `-`, `from_f32`, `from_i32`, `checked_div` and `fixed_dot` all clamp to `[Q16::MIN, Q16::MAX]` rather than wrapping, because a wrapped sign flip inside a recurrence is indistinguishable from a legitimate value. Multiplication and `fixed_dot` round to nearest so repeated products carry no one-sided bias.

See `examples/fixed_point_cortex_m.rs` for the full version, which runs 32 steps in both precisions and reports the worst divergence (`~1e-5` on the shipped parameters).

## Memory

No cycle-accurate timing benchmarks are published for this crate (there is no `benches/` suite yet). The figures below are **derived from the code**: `SsmState` allocates `d_state + d_inner` floats, `Q16SsmState` allocates `d_state` words. The test `presets::tests::test_documented_state_ram_matches_allocation` keeps this table honest.

| Preset | `d_model` | `d_state` | `d_inner` | `SsmState` (`h` + `prev_x`) | `Q16SsmState` (`h`) |
|---|---|---|---|---|---|
| `stm32h7` | 64 | 16 | 128 | 576 B | 64 B |
| `rp2040` | 32 | 8 | 64 | 288 B | 32 B |
| `esp32c3` | 48 | 12 | 96 | 432 B | 48 B |

- Only the owned state constructors allocate; the per-step recurrence performs no heap allocation, and the `*_slice` kernels let the caller place the state in `.bss` instead.
- The Q16 path carries no convolution history, so it needs `h` only.
- **Weight sizes are deliberately not quoted.** This crate holds no projection weights and no block structure — only the diagonal recurrence — so any "KB of INT8 weights per block" figure here would describe a model the crate cannot see. Size flash from the checkpoint you actually deploy.
- `S4Step` carries a genuine complex state (`h_re` + `h_im`, `2 * d_state` floats) so that the imaginary pole component actually produces oscillatory modes. Its output is `Re(C·h)`, not the conjugate-pair `2·Re(C·h)`, so that with `lambda_im = 0` it reproduces the host `kizzasi-model/src/s4.rs` output exactly; fold the factor of two into `C` if your weights assume otherwise.

## Documentation

- Generate API docs locally: `cargo doc -p kizzasi-embedded --all-features --open`
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under the Apache License, Version 2.0.
