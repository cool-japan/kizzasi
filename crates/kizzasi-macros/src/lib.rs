//! Procedural macros for Kizzasi AGSP.
//!
//! This crate provides derive macros to simplify working with Kizzasi predictors
//! and custom configurations.
//!
//! # Derive Macros
//!
//! ## `#[derive(KizzasiConfig)]`
//!
//! Automatically implements a builder pattern (`<Type>::builder()`) and
//! per-field validation for custom configurations. `build()` returns
//! `Result<Self, <Type>BuilderError>`.
//!
//! ```rust
//! use kizzasi_macros::KizzasiConfig;
//!
//! fn validate_dim(dim: &usize) -> Result<(), String> {
//!     if *dim > 0 && *dim % 64 == 0 {
//!         Ok(())
//!     } else {
//!         Err("Dimension must be positive and divisible by 64".into())
//!     }
//! }
//!
//! #[derive(KizzasiConfig)]
//! struct MyCustomConfig {
//!     #[config(default = 4096)]
//!     context_window: usize,
//!
//!     #[config(validate = "validate_dim")]
//!     hidden_dim: usize,
//!
//!     learning_rate: f64,
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = MyCustomConfig::builder()
//!     .hidden_dim(128)
//!     .learning_rate(1e-3)
//!     .build()?;
//! assert_eq!(config.context_window, 4096); // used the default
//! assert_eq!(config.hidden_dim, 128);
//! #     Ok(())
//! # }
//! ```
//!
//! ## `#[derive(Preset)]`
//!
//! Generates preset constructor functions for common configurations.
//!
//! ```rust
//! use kizzasi_macros::Preset;
//!
//! #[derive(Preset)]
//! #[preset(name = "audio", context_window = 8192, hidden_dim = 256)]
//! #[preset(name = "video", context_window = 16384, hidden_dim = 512)]
//! struct ModelConfig {
//!     context_window: usize,
//!     hidden_dim: usize,
//! }
//!
//! // Generated:
//! // impl ModelConfig {
//! //     pub fn audio_preset() -> Self { ... }
//! //     pub fn video_preset() -> Self { ... }
//! // }
//!
//! let audio = ModelConfig::audio_preset();
//! let video = ModelConfig::video_preset();
//! assert_eq!(audio.context_window, 8192);
//! assert_eq!(video.hidden_dim, 512);
//! ```
//!
//! ## `#[derive(Instrumented)]`
//!
//! Implements `kizzasi::telemetry::Instrumented` for the struct, exposing an
//! `Arc<MetricsCollector>` accessor resolved from a `#[metrics]`-annotated
//! field (or a field named `collector` of type `Arc<...>`). It does **not**
//! wrap or time any method — see the macro's own documentation below for
//! details. Requires the consuming crate to depend on `kizzasi` directly.
//!
//! ```rust,ignore
//! use kizzasi::telemetry::MetricsCollector;
//! use kizzasi_macros::Instrumented;
//! use std::sync::Arc;
//!
//! #[derive(Instrumented)]
//! struct MyPredictor {
//!     #[metrics]
//!     collector: Arc<MetricsCollector>,
//! }
//! ```

extern crate proc_macro;
use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput};

mod attrs;
mod config;
mod instrumented;
mod preset;

/// Derive macro for custom Kizzasi configurations.
///
/// Generates a builder pattern (`<Type>::builder()` / `<Type>Builder`) with
/// automatic validation and default values. The generated builder and its
/// methods mirror the annotated struct's own visibility (a `pub(crate)`
/// config struct gets a `pub(crate)` builder, not an unconditionally `pub`
/// one). Supports generic structs, including lifetime parameters.
///
/// # Attributes
///
/// - `#[config(default = value)]` — Makes a field optional in the builder,
///   falling back to `value` (any Rust expression) when unset. `value` may
///   also be written as a *quoted* string, which is reinterpreted as source
///   (mirroring `validate = "path"` below) — useful for expressions that are
///   awkward to write unquoted. That quoted form must not be used to spell a
///   literal string value: a bare single-identifier result (e.g.
///   `default = "None"`) is rejected as ambiguous between "the path `None`"
///   and "the text `None`" — write it unquoted (`default = None`) if you
///   meant the path, or escape actual text as source, e.g.
///   `default = "\"hello\""` for `&str` or `default = "\"hello\".to_string()"`
///   for `String`.
/// - `#[config(validate = "function")]` — Runs `fn(&T) -> Result<(), String>`
///   against the built value; a returned `Err` is wrapped into
///   `<Type>BuilderError::Validation { field, message }`.
/// - `#[config(skip)]` — Excludes the field from the builder entirely; it is
///   filled from `#[config(default = ...)]` if present, or
///   `Default::default()` otherwise.
///
/// A field of type `Option<T>` with neither `#[config(default = ...)]` nor
/// `#[config(skip)]` is automatically optional: the builder setter takes the
/// unwrapped `T`, and simply never calling it builds to `None`.
///
/// `build()` returns `Result<Self, <Type>BuilderError>`. `BuilderError`
/// implements `std::error::Error` + `Display`, and `impl From<BuilderError>
/// for String` so `?` still composes with an existing
/// `Result<_, String>`-returning function.
///
/// # Example
///
/// ```rust
/// use kizzasi_macros::KizzasiConfig;
///
/// fn validate_positive(v: &f64) -> Result<(), String> {
///     if *v > 0.0 { Ok(()) } else { Err("must be positive".into()) }
/// }
///
/// #[derive(KizzasiConfig)]
/// struct MyConfig {
///     #[config(default = 1024)]
///     buffer_size: usize,
///
///     #[config(validate = "validate_positive")]
///     sample_rate: f64,
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = MyConfig::builder().sample_rate(48_000.0).build()?;
/// assert_eq!(config.buffer_size, 1024);
/// #     Ok(())
/// # }
/// ```
#[proc_macro_derive(KizzasiConfig, attributes(config))]
pub fn derive_kizzasi_config(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    config::expand(input)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Derive macro for generating preset constructors.
///
/// Supports generic structs, including lifetime parameters.
///
/// # Attributes
///
/// - `#[preset(name = "preset_name", field1 = value1, field2 = value2, ...)]`
///
/// Each preset attribute generates a static constructor method named
/// `<preset_name>_preset`. `name` must be a non-empty snake_case identifier
/// (it is spliced directly into the generated function name). Stacking
/// multiple `#[preset(...)]` attributes on the same struct produces multiple
/// constructors; a preset that does not set every field falls back to
/// `..Default::default()` for the rest (which requires the struct to
/// implement `Default`; a preset that *does* set every field does not).
/// `#[derive(Preset)]` requires at least one `#[preset(...)]` attribute.
///
/// # Example
///
/// ```rust
/// use kizzasi_macros::Preset;
///
/// #[derive(Preset)]
/// #[preset(name = "fast", workers = 4, buffer = 1024)]
/// #[preset(name = "balanced", workers = 8, buffer = 4096)]
/// struct Config {
///     workers: usize,
///     buffer: usize,
/// }
///
/// let config = Config::fast_preset();
/// assert_eq!(config.workers, 4);
/// assert_eq!(config.buffer, 1024);
/// ```
#[proc_macro_derive(Preset, attributes(preset))]
pub fn derive_preset(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    preset::expand(input)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Derive macro that implements `kizzasi::telemetry::Instrumented` for a
/// struct.
///
/// This macro does **not** wrap, time, or otherwise instrument any method —
/// a derive macro cannot rewrite inherent `impl` blocks written elsewhere in
/// the crate. What it generates is exactly one thing: an implementation of
/// the `Instrumented` trait whose `metrics()` method returns a clone of the
/// struct's `Arc<MetricsCollector>`, so the struct can be handed to any code
/// that is generic over `Instrumented`.
///
/// The collector field is resolved as:
/// - the field annotated `#[metrics]`, if there is exactly one, or
/// - otherwise, a field literally named `collector` whose type's last path
///   segment is `Arc` (e.g. `Arc<MetricsCollector>`) — a same-named field of
///   an unrelated type is not matched.
///
/// If neither is found, expansion fails with a compile error. Supports
/// generic structs.
///
/// Because the generated code refers to `kizzasi::telemetry::{Instrumented,
/// MetricsCollector}`, the crate using this derive must depend on `kizzasi`
/// directly; a rename via `package = "kizzasi"` in `Cargo.toml` is resolved
/// automatically.
///
/// # Example
///
/// ```rust,ignore
/// use kizzasi::telemetry::MetricsCollector;
/// use kizzasi_macros::Instrumented;
/// use std::sync::Arc;
///
/// #[derive(Instrumented)]
/// struct MyPredictor {
///     #[metrics]
///     collector: Arc<MetricsCollector>,
/// }
///
/// // Generated:
/// // impl kizzasi::telemetry::Instrumented for MyPredictor {
/// //     fn metrics(&self) -> Arc<MetricsCollector> { self.collector.clone() }
/// // }
/// ```
#[proc_macro_derive(Instrumented, attributes(metrics))]
pub fn derive_instrumented(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    instrumented::expand(input)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}
