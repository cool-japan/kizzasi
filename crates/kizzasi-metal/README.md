# kizzasi-metal

Apple Metal backend activation shim for the Kizzasi ecosystem.

![version](https://img.shields.io/badge/version-0.2.4-blue)
![status](https://img.shields.io/badge/status-alpha-orange)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

## Overview

This crate exists to solve a Cargo problem, not a Rust one.

`kizzasi-core` exposes a `metal` feature that is meaningful only on Apple
hardware. A Cargo **feature**, however, cannot be made conditional on the
target: writing

```toml
metal = ["candle-core/metal"]   # what kizzasi-core used to do
```

turns candle's Metal backend on for Linux and Windows builds as well, and that
backend depends on `objc2`, whose `lib.rs` opens with

```
error: `objc2` only works on Apple platforms.
```

The practical fallout was that `cargo build --all-features` and
`cargo test --all-features` — the commands the README documents and CI runs —
could not complete anywhere except macOS.

A Cargo **dependency**, unlike a feature, *can* be target-scoped, and under
`resolver = "2"` the features requested by a platform-specific dependency are
ignored for targets that are not currently being built. So this crate declares
`candle-core` twice:

```toml
[target.'cfg(target_vendor = "apple")'.dependencies]
candle-core = { workspace = true, features = ["metal"] }
candle-nn = { workspace = true, features = ["metal"] }

[target.'cfg(not(target_vendor = "apple"))'.dependencies]
candle-core.workspace = true
```

and `kizzasi-core`'s `metal` feature becomes `metal = ["dep:kizzasi-metal"]`.
Apple builds light up the real GPU backend through candle; every other target
builds a small inert crate and `--all-features` works again.

### Why not a renamed alias?

Keeping the second declaration inside `kizzasi-core` as a target-scoped
`package = "candle-core"` alias was tried first and rejected by Cargo itself:
`cargo metadata --filter-platform` refuses a manifest that "depends on crate
candle-core v0.11.0 multiple times with different names", which in turn breaks
`cargo-nextest` and `cargo-deny`. Moving the second declaration into its own
package keeps every manifest single-named and every tool happy.

## Inert, not silent

On a non-Apple target the `metal` feature is buildable but cannot produce a GPU
device — and it says so rather than degrading quietly:

| API | Apple target | Other targets |
| --- | --- | --- |
| `BACKEND_COMPILED` | `true` | `false` |
| `unavailable_reason()` | `None` | `Some("… only for Apple targets …")` |
| `new_device(n)` | candle's `Device::new_metal(n)` | `Err` naming the target |
| `is_available()` | whether a device answers | `false` |
| `device_ordinals()` | `[0]` when present | `[]` |

Nothing here hands back a CPU device labelled as a GPU.

## Usage

Depend on it through `kizzasi-core` rather than directly:

```toml
[dependencies]
kizzasi-core = { version = "0.2", features = ["metal"] }
```

Direct use, if you need to probe the backend yourself:

```rust
match kizzasi_metal::new_device(0) {
    Ok(device) => println!("Metal device ready: {device:?}"),
    Err(err) => println!("no Metal device: {err}"),
}
```

## Purity

**NOTE: FFI.** On Apple targets this crate transitively links Apple's Metal
framework through `objc2`. That is why the whole path stays opt-in behind
`kizzasi-core/metal`; the default Kizzasi build is pure Rust.

## License

Apache-2.0
