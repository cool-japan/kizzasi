//! Procedural macros for Kizzasi AGSP.
//!
//! This crate provides derive macros to simplify working with Kizzasi predictors
//! and custom configurations.
//!
//! # Derive Macros
//!
//! ## `#[derive(KizzasiConfig)]`
//!
//! Automatically implements builder pattern and validation for custom configurations.
//!
//! ```rust,ignore
//! use kizzasi_macros::KizzasiConfig;
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
//! fn validate_dim(dim: &usize) -> Result<(), String> {
//!     if *dim > 0 && *dim % 64 == 0 {
//!         Ok(())
//!     } else {
//!         Err("Dimension must be positive and divisible by 64".into())
//!     }
//! }
//! ```
//!
//! ## `#[derive(Preset)]`
//!
//! Generates preset constructor functions for common configurations.
//!
//! ```rust,ignore
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
/// Generates a builder pattern with automatic validation and default values.
///
/// # Attributes
///
/// - `#[config(default = value)]` - Set a default value for a field
/// - `#[config(validate = "function")]` - Specify a validation function
/// - `#[config(skip)]` - Skip this field in the builder
///
/// # Example
///
/// ```rust,ignore
/// #[derive(KizzasiConfig)]
/// struct MyConfig {
///     #[config(default = 1024)]
///     buffer_size: usize,
///
///     #[config(validate = "validate_positive")]
///     sample_rate: f64,
/// }
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
/// # Attributes
///
/// - `#[preset(name = "preset_name", field1 = value1, field2 = value2, ...)]`
///
/// Each preset attribute generates a static constructor method.
///
/// # Example
///
/// ```rust,ignore
/// #[derive(Preset)]
/// #[preset(name = "fast", workers = 4, buffer = 1024)]
/// #[preset(name = "balanced", workers = 8, buffer = 4096)]
/// struct Config {
///     workers: usize,
///     buffer: usize,
/// }
///
/// // Usage:
/// let config = Config::fast_preset();
/// ```
#[proc_macro_derive(Preset, attributes(preset))]
pub fn derive_preset(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    preset::expand(input)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Derive macro for metrics instrumentation.
///
/// Automatically adds metrics collection to struct methods.
///
/// # Example
///
/// ```rust,ignore
/// #[derive(Instrumented)]
/// struct MyPredictor {
///     #[metrics]
///     collector: Arc<MetricsCollector>,
/// }
/// ```
#[proc_macro_derive(Instrumented, attributes(metrics))]
pub fn derive_instrumented(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    instrumented::expand(input)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}
