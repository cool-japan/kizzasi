# kizzasi-macros

Procedural macros for the Kizzasi AGSP ecosystem.

![status](https://img.shields.io/badge/status-stable-brightgreen)
![version](https://img.shields.io/badge/version-0.2.4-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

## Overview

This crate provides derive macros and procedural macro helpers for Kizzasi, enabling ergonomic configuration patterns and reducing boilerplate code. All three derive macros are fully implemented with no stub code; 80 tests pass in this crate (`cargo nextest run -p kizzasi-macros --all-features`), covering every documented behavior of `KizzasiConfig` and `Preset`, including every rejection path (invalid names, duplicate keys, name collisions, generics, and so on), via a mix of end-to-end `tests/*.rs` integration tests and `#[cfg(test)]` unit tests that exercise the macros' internal `syn::Result`-returning `expand`/`parse_*` functions directly. Four of `lib.rs`'s five doc examples are real, `cargo test --doc`-checked doctests; the fifth (`#[derive(Instrumented)]`) stays `rust,ignore` since it needs a real dependency on `kizzasi`, which would be circular from this crate. `#[derive(Instrumented)]`'s error paths and crate-path resolution are unit-tested the same way from within this crate; its successful, end-to-end expansion is tested from the `kizzasi` crate, whose `telemetry` module its generated code depends on.

## Features

- `#[derive(KizzasiConfig)]` — Generates a builder pattern (`<Type>::builder()` / `<Type>Builder`) with per-field validation. The builder and its methods mirror the annotated struct's own visibility. Field attributes: `#[config(default = expr)]` makes a field optional in the builder with a fallback value, `#[config(validate = "fn_path")]` runs a `fn(&T) -> Result<(), String>` check against the built value, and `#[config(skip)]` excludes a field from the builder entirely (filled from its default expression, or `Default::default()` if none is given). A field of type `Option<T>` with neither attribute is automatically optional: the setter takes the unwrapped `T`, and not calling it builds to `None`. `build()` returns `Result<Self, <Type>BuilderError>` — a per-type error implementing `std::error::Error` + `Display` (and `impl From<BuilderError> for String`, so `?` still composes with an existing `Result<_, String>`-returning function).
- `#[derive(Preset)]` — Generates named `<name>_preset() -> Self` constructors from one or more `#[preset(name = "...", field = value, ...)]` attributes. `name` must be a non-empty snake_case identifier (it becomes part of the generated function name; other characters, leading digits, and non-snake_case names are rejected with a spanned compile error instead of a panic or a silent `non_snake_case` warning). Multiple `#[preset(...)]` attributes can be stacked on the same struct to produce multiple constructors; a preset that doesn't set every field falls back to `..Default::default()` for the rest (needing `Self: Default`, required only on the methods that actually need it — a full-coverage preset does not force a `Default` bound onto an otherwise non-`Default` generic struct). Duplicate field keys or a duplicate `name` within one `#[preset(...)]` are rejected; `#[derive(Preset)]` with no `#[preset(...)]` attribute at all is rejected rather than silently expanding to an empty `impl`.
- `#[derive(Instrumented)]` — Implements `kizzasi::telemetry::Instrumented` for the struct. Looks for a field annotated `#[metrics]`, falling back to a field literally named `collector` whose type's last path segment is `Arc` (a same-named field of an unrelated type is not matched). The generated code resolves the `kizzasi` crate path at expansion time (via `proc-macro-crate`), so it keeps working if the dependency is renamed in `Cargo.toml` (`agsp = { package = "kizzasi" }`); it does not wrap, time, or otherwise instrument any method — it only implements the trait accessor.

All three macros support generic structs, including lifetime parameters; all three still require named fields (no tuple/unit structs).

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
kizzasi-macros = "0.2.4"
```

### `#[derive(KizzasiConfig)]`

```rust
use kizzasi_macros::KizzasiConfig;

fn validate_positive(v: &usize) -> Result<(), String> {
    if *v > 0 { Ok(()) } else { Err("must be positive".into()) }
}

#[derive(KizzasiConfig)]
struct MyConfig {
    #[config(default = 4096)]
    context_window: usize,
    #[config(validate = "validate_positive")]
    batch_size: usize,
    learning_rate: f32,
}

// context_window falls back to 4096 when unset; batch_size is checked by
// validate_positive when build() runs; learning_rate has no attributes,
// so it stays required. build() returns Result<MyConfig, MyConfigBuilderError>;
// `?` still works in a function returning `Result<_, String>` via the
// generated `impl From<MyConfigBuilderError> for String`.
let config = MyConfig::builder()
    .batch_size(32)
    .learning_rate(0.001)
    .build()?;
```

`#[config(default = "...")]` also accepts a *quoted* expression, reinterpreted
as Rust source (mirroring `validate = "path"`) — useful for expressions that
are awkward to write unquoted. That form must not be used to spell a literal
string value: a bare single-identifier result (e.g. `default = "None"`) is
rejected as ambiguous between "the path `None`" and "the text `None`" — write
it unquoted (`default = None`) if you meant the path, or escape actual text as
source, e.g. `default = "\"hello\""` for `&str` or
`default = "\"hello\".to_string()"` for `String`.

### `#[derive(Preset)]`

```rust
use kizzasi_macros::Preset;

#[derive(Preset)]
#[preset(name = "audio", context_window = 8192, hidden_dim = 256)]
#[preset(name = "video", context_window = 16384, hidden_dim = 512)]
struct ModelConfig {
    context_window: usize,
    hidden_dim: usize,
}

let audio = ModelConfig::audio_preset();
let video = ModelConfig::video_preset();
assert_eq!(audio.context_window, 8192);
assert_eq!(video.hidden_dim, 512);
```

### `#[derive(Instrumented)]`

The generated `impl` refers to `kizzasi::telemetry::Instrumented` and
`kizzasi::telemetry::MetricsCollector` by a path resolved at expansion time,
so the consuming crate must depend on `kizzasi` directly (a rename via
`package = "kizzasi"` in `Cargo.toml` is resolved automatically):

```rust,ignore
use kizzasi::telemetry::MetricsCollector;
use kizzasi_macros::Instrumented;
use std::sync::Arc;

#[derive(Instrumented)]
struct MyPredictor {
    #[metrics]
    collector: Arc<MetricsCollector>,
}
```

## Documentation

- [API Documentation](https://docs.rs/kizzasi-macros)
- [Kizzasi Repository](https://github.com/cool-japan/kizzasi)

## License

Licensed under the Apache License, Version 2.0.
